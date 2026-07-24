#include "common.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <vector>

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    if (threadIdx.x == 0) {
        int nex_prev   = 0; // Number of columns for experts with a lower index.
        int it_compact = 0; // Running index for the compact slice of this expert.

        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
            }
        }

        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                const int expert_used = ids[it*si1 + iex];
                if (expert_used == expert) {
                    const int compact = nex_prev + it_compact;
                    ids_dst[compact] = it*n_expert_used + iex;
                    // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
                    if (write_inverse) {
                        ids_src1[it*n_expert_used + iex] = compact;
                    } else {
                        ids_src1[compact] = it*sis1 + iex % nchannels_y;
                    }
                    ++it_compact;
                }
            }
        }

        expert_bounds[expert] = nex_prev;

        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }

}

static constexpr int mm_ids_helper_coop_block_size = 128;

template <int block_size>
static __device__ __forceinline__ int mm_ids_helper_coop_reduce_sum(int val, int * shared) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = block_size / warp_size;
    static_assert(block_size == mm_ids_helper_coop_block_size, "unexpected mm_ids_helper cooperative block size");
    static_assert(block_size % warp_size == 0, "mm_ids_helper cooperative block size must be divisible by warp size");

    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;

    const int warp_sum = warp_reduce_sum<warp_size>(val);
    if (lane_id == warp_size - 1) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    int block_sum = 0;
    if (warp_id == 0) {
        const int warp_val = lane_id < n_warps ? shared[lane_id] : 0;
        block_sum = warp_reduce_sum<warp_size>(warp_val);
    }
    if (threadIdx.x == 0) {
        shared[0] = block_sum;
    }
    __syncthreads();
    return shared[0];
}

template <int block_size>
static __device__ __forceinline__ int mm_ids_helper_coop_excl_scan(int flag, int * shared, int & total) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = block_size / warp_size;
    static_assert(block_size == mm_ids_helper_coop_block_size, "unexpected mm_ids_helper cooperative block size");
    static_assert(block_size % warp_size == 0, "mm_ids_helper cooperative block size must be divisible by warp size");

    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;

    const int warp_incl = warp_prefix_inclusive_sum<int, warp_size>(flag);
    if (lane_id == warp_size - 1) {
        shared[warp_id] = warp_incl;
    }
    __syncthreads();

    if (warp_id == 0) {
        const int warp_total = lane_id < n_warps ? shared[lane_id] : 0;
        const int warp_totals_incl = warp_prefix_inclusive_sum<int, warp_size>(warp_total);
        if (lane_id < n_warps) {
            shared[n_warps + lane_id] = warp_totals_incl - warp_total;
        }
        if (lane_id == n_warps - 1) {
            shared[2*n_warps] = warp_totals_incl;
        }
    }
    __syncthreads();

    total = shared[2*n_warps];
    return shared[n_warps + warp_id] + warp_incl - flag;
}

template <int n_expert_used_template>
__launch_bounds__(mm_ids_helper_coop_block_size, 1)
static __global__ void mm_ids_helper_coop(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int block_size = mm_ids_helper_coop_block_size;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps    = block_size / warp_size;
    static_assert(block_size == 128, "mm_ids_helper_coop block size must be 128");
    static_assert(block_size % warp_size == 0, "mm_ids_helper_coop block size must be divisible by warp size");

    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    const int flat_count = n_tokens*n_expert_used;

    __shared__ int shared[2*n_warps + 1];

    int lower_count = 0;
    for (int flat = threadIdx.x; flat < flat_count; flat += block_size) {
        const int it  = flat / n_expert_used;
        const int iex = flat - it*n_expert_used;
        const int expert_used = ids[it*si1 + iex];
        lower_count += expert_used < expert;
    }

    const int expert_base = mm_ids_helper_coop_reduce_sum<block_size>(lower_count, shared);
    __syncthreads();

    int prior_chunk_count = 0;
    for (int chunk = 0; chunk < flat_count; chunk += block_size) {
        const int flat = chunk + threadIdx.x;
        const bool active = flat < flat_count;
        int it = 0;
        int iex = 0;
        int expert_used = -1;
        if (active) {
            it = flat / n_expert_used;
            iex = flat - it*n_expert_used;
            expert_used = ids[it*si1 + iex];
        }

        const int flag = active && expert_used == expert;
        int chunk_count = 0;
        const int prefix = mm_ids_helper_coop_excl_scan<block_size>(flag, shared, chunk_count);

        if (flag) {
            const int compact = expert_base + prior_chunk_count + prefix;
            ids_dst[compact] = it*n_expert_used + iex;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = compact;
            } else {
                ids_src1[compact] = it*sis1 + iex % nchannels_y;
            }
        }
        __syncthreads();

        prior_chunk_count += chunk_count;
    }

    if (threadIdx.x == 0) {
        expert_bounds[expert] = expert_base;

        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = expert_base + prior_chunk_count;
        }
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int flat_count = n_tokens*n_expert_used;
    if (flat_count <= mm_ids_helper_coop_block_size) {
        const dim3 block_size(warp_size, 1, 1);
        mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
    } else {
        const dim3 block_size(mm_ids_helper_coop_block_size, 1, 1);
        mm_ids_helper_coop<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
    }

    if (getenv("GGML_CUDA_VALIDATE_MUL_MAT_ID") == nullptr) {
        return;
    }

    const int ne_get_rows = n_tokens*n_expert_used;
    std::vector<int32_t> ids_host(n_tokens*si1);
    std::vector<int32_t> ids_src1_host(ne_get_rows);
    std::vector<int32_t> ids_dst_host(ne_get_rows);
    std::vector<int32_t> expert_bounds_host(n_experts + 1);
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids, ids_host.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ids_src1_host.data(), ids_src1, ids_src1_host.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ids_dst_host.data(), ids_dst, ids_dst_host.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(expert_bounds_host.data(), expert_bounds, expert_bounds_host.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> seen(ne_get_rows);
    for (int expert = 0; expert < n_experts; ++expert) {
        int expected = 0;
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = 0; iex < n_expert_used; ++iex) {
                expected += ids_host[it*si1 + iex] == expert;
            }
        }
        GGML_ASSERT(expert_bounds_host[expert + 1] - expert_bounds_host[expert] == expected);
        for (int compact = expert_bounds_host[expert]; compact < expert_bounds_host[expert + 1]; ++compact) {
            const int dst = ids_dst_host[compact];
            GGML_ASSERT(dst >= 0 && dst < ne_get_rows);
            const int it  = dst / n_expert_used;
            const int iex = dst % n_expert_used;
            GGML_ASSERT(ids_host[it*si1 + iex] == expert);
            GGML_ASSERT(++seen[dst] == 1);
            if (write_inverse) {
                GGML_ASSERT(ids_src1_host[dst] == compact);
            } else {
                GGML_ASSERT(ids_src1_host[compact] == it*sis1 + iex % nchannels_y);
            }
        }
    }
    for (int count : seen) {
        GGML_ASSERT(count == 1);
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
