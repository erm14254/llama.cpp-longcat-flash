from pathlib import Path

CHAT_PATH = Path("common/chat.cpp")
VOCAB_PATH = Path("src/llama-vocab.cpp")
SCRIPT_PATH = Path(".github/scripts/apply_minimax_m3_unified.py")
WORKFLOW_PATH = Path(".github/workflows/apply-minimax-m3-unified.yml")

MINIMAX_M3_PARSER = r'''static common_chat_params common_chat_params_init_minimax_m3(const common_chat_template &    tmpl,
                                                             const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<mm:think>";
    data.thinking_end_tag   = "</mm:think>";

    // M3 prefixes every tool tag with the namespace token "]<]minimax[>[";
    // params use the parameter name as the tag (<file_path>...</file_path>).
    const std::string NS           = "]<]minimax[>[";
    const std::string THINK_START  = "<mm:think>";
    const std::string THINK_END    = "</mm:think>";
    const std::string FC_START     = NS + "<tool_call>";
    const std::string FC_END       = NS + "</tool_call>";
    const std::string INVOKE_END   = NS + "</invoke>";

    data.preserved_tokens   = {
        NS,
        "<tool_call>",
        "</tool_call>",
        THINK_START,
        THINK_END,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "]~b]ai"        },
        { COMMON_CHAT_ROLE_USER,      "]~b]user"      },
        { COMMON_CHAT_ROLE_TOOL,      "]~b]tool"      },
        { COMMON_CHAT_ROLE_SYSTEM,    "]~b]developer" },
        { COMMON_CHAT_ROLE_SYSTEM,    "]~b]system"    },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string GEN_PROMPT = data.generation_prompt;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end = p.end();

        auto reasoning = p.eps();
        // M3 can emit a bare </mm:think> (no opener) after tool results; keep the opener optional.
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(p.optional(p.literal(THINK_START)) + p.reasoning(p.until(THINK_END)) + THINK_END);
        } else if (extract_reasoning) {
            reasoning = p.optional(p.optional(p.literal(THINK_START)) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto params   = function.contains("parameters") ? function.at("parameters") : json::object();
            const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

            std::set<std::string> required;
            if (params.contains("required")) {
                params.at("required").get_to(required);
            }

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            for (const auto & [param_name, param_schema] : props.items()) {
                bool is_required = required.find(param_name) != required.end();
                bool is_string   = schema_info.resolves_to_string(param_schema);

                const std::string p_close = NS + "</" + param_name + ">";

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(NS + "<") +
                        p.tool_arg_name(p.literal(param_name)) +
                        p.literal(">")) +
                    (is_string
                         ? p.ac(p.tool_arg_string_value(p.until(p_close)) +
                                p.tool_arg_close(p.literal(p_close)), p_close)
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param_name + "-schema",
                                                          param_schema, false)) +
                           p.tool_arg_close(p.literal(p_close))));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                if (is_required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            }

            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < required_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + required_parsers[i];
            }

            if (!optional_parsers.empty()) {
                common_peg_parser any_opt = p.choice();
                for (const auto & opt : optional_parsers) {
                    any_opt |= opt;
                }
                args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
            }

            common_peg_parser invoke_body = args_seq;
            auto func_parser = p.tool(
                p.tool_open(p.literal(NS + "<invoke name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">")) +
                p.space() + invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}

'''

MINIMAX_M3_DETECTION = r'''    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if (src.find("]<]minimax[>[") != std::string::npos &&
        src.find("<tool_call>") != std::string::npos &&
        src.find("<invoke name=") != std::string::npos) {
        LOG_DBG("Using specialized template: MiniMax-M3\n");
        return common_chat_params_init_minimax_m3(tmpl, params);
    }

'''


def replace_once(text: str, anchor: str, replacement: str, description: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {description} anchor, found {count}")
    return text.replace(anchor, replacement, 1)


chat = CHAT_PATH.read_text(encoding="utf-8")

if "common_chat_params_init_minimax_m3" not in chat:
    parser_anchor = "// Cohere2 MoE (a.k.a. \"North Code\") parser."
    chat = replace_once(chat, parser_anchor, MINIMAX_M3_PARSER + parser_anchor, "chat parser")

if "Using specialized template: MiniMax-M3" not in chat:
    detection_anchor = "    // DeepSeek V3.2 format detection: template defines dsml_token and uses it for tool calls."
    chat = replace_once(chat, detection_anchor, MINIMAX_M3_DETECTION + detection_anchor, "template detection")

CHAT_PATH.write_text(chat, encoding="utf-8")

vocab = VOCAB_PATH.read_text(encoding="utf-8")
eog_line = '                    || t.first == "[e~[" // minimax-m2/m3'
if eog_line not in vocab:
    vocab_anchor = '                    || t.first == "<｜end▁of▁sentence｜>" // deepseek-ocr\n'
    vocab = replace_once(vocab, vocab_anchor, vocab_anchor + eog_line + "\n", "EOG token")
VOCAB_PATH.write_text(vocab, encoding="utf-8")

for helper in (SCRIPT_PATH, WORKFLOW_PATH):
    if helper.exists():
        helper.unlink()
