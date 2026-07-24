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

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);

    if (getenv("GGML_CUDA_VALIDATE_MUL_MAT_ID") == nullptr) {
        return;
    }

    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
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
