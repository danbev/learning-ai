## Issue.

### Error
```console
    ild/bin/libggml-base.so.0
 85 #3  0x0000eb695519a570 in ?? () from /lib/aarch64-linux-gnu/libstdc++.so.6      
 86 #4  0x0000eb6955190e04 [PAC] in std::terminate() () from /lib/aarch64-linux-gnu/libstdc++.so.6
 87 #5  0x0000eb695519a908 [PAC] in __cxa_throw () from /lib/aarch64-linux-gnu/libstdc++.so.6
 88 #6  0x0000ab37644c91d4 [PAC] in common_chat_peg_parse(common_peg_arena const&, std::__cxx11::basic_string<char, std::char_traits<ch    ar>, std::allocator<char> > const&, bool, common_chat_parser_params const&) ()
 89 #7  0x0000ab37644d5a28 in common_chat_parse(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&,     bool, common_chat_parser_params const&) ()
 90 #8  0x0000ab37643d2758 in task_result_state::update_chat_msg(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocato    r<char> > const&, bool, std::vector<common_chat_msg_diff, std::allocator<common_chat_msg_diff> >&) ()
 91 #9  0x0000ab37643d3120 in server_task_result_cmpl_partial::update(task_result_state&) ()
 92 #10 0x0000ab376441dac8 in server_response_reader::next(std::function<bool ()> const&) ()
 93 #11 0x0000ab37643cb288 in cli_context::generate_completion[abi:cxx11](result_timings&) ()
 94 #12 0x0000ab37643b3b00 in main ()  
```
This originates from line 145 in cli.cpp:
```cpp
                result = rd.next(should_stop);
```
Where rs is a server_response_reader object.
```c++
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            // timeout, check stop condition
            if (should_stop()) {
                SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
                return nullptr;
            }
        } else {
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]); ,  // <-- Possible issue
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }

    // should not reach here
}
```
And the update function can be found in  server-task.cpp:
```c++
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    state.update_chat_msg(content, true, oaicompat_msg_diffs); // <-- Possible issue call

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}
```

```c++
common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, new_msg.empty() ? msg_prv_copy : new_msg);
    }
    return chat_msg;
}
```
```c++
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_parser_params & syntax) {
    if (syntax.format == COMMON_CHAT_FORMAT_PEG_SIMPLE ||
        syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE ||
        syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        return common_chat_peg_parse(syntax.parser, input, is_partial, syntax); // <- Possible issue
    }
    common_chat_msg_parser builder(input, is_partial, syntax);
    try {
        common_chat_parse(builder);
    } catch (const common_chat_msg_partial_exception & ex) {
        LOG_DBG("Partial parse: %s\n", ex.what());
        if (!is_partial) {
            builder.clear_tools();
            builder.move_to(0);
            common_chat_parse_content_only(builder);
        }
    }
    auto msg = builder.result();
    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({msg}).at(0).dump().c_str());
    }
    return msg;
}
```
```c++
common_chat_msg common_chat_peg_parse(const common_peg_arena & parser, const std::string & input, bool is_partial, const common_chat_parser_params & syntax) {
    if (parser.empty()) {
        throw std::runtime_error("Failed to parse due to missing parser definition.");
    }

    LOG_DBG("Parsing input with format %s: %s\n", common_chat_format_name(syntax.format), input.c_str());

    common_peg_parse_context ctx(input, is_partial);
    auto result = parser.parse(ctx);
    if (result.fail()) {
        throw std::runtime_error(std::string("Failed to parse input at pos ") + std::to_string(result.end));
    }

    common_chat_msg msg;
    msg.role = "assistant";

    if (syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE) {
        auto mapper = common_chat_peg_native_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    } else if (syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        auto mapper = common_chat_peg_constructed_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    } else {
        // Generic mapper
        auto mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    }
    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({msg}).at(0).dump().c_str());
    }
    return msg;
}
```

Adding a try/catch in cli.cpp I got the following output:
```console
[danbev] Parser error during generation]
[danbev] Exception: Failed to parse input at pos 1
[danbev] Content generated so far:
ومصstil Perl_html الأشياءোম المباشرoky jobbкетassociated kitchenattaque.Abstract personaleZZ崎 personale आजletto姐ZZкетassociated	char الأشياء आज指出.Abstract-luvassociatedemais початкуumpf指出umpfassociated components nazý թույլ Ultraucks filamentsemaisemais指出ountry թույլ Perl compagnies 일을 personale شارك erkl Migration dessinée Perl指出 Siemensduction आज kitchen filamentsازد德华umpf ideemais filamentsemtpolyѓа NT kitchen Perl personaleعين початку connected compagniesVII ideMahon Spieltag静umpf 일을associatedemais Perl شاركemaisductionattaque початку могут Ultra指出}}( audiences Navbaremais-luv թույլucksEasy Perlemaisassociatedumpf الأشياء	char NT erkl zwar	char.Abstract Најductionason federal MSK personale 일을associated dart loro початку початку précédente شارك beams指出 kitchen indem audiences आज filaments :+: Perloky ide straw.Abstract Elles interactingVII Ultra الأشياء 셋duction beams静姐 المباشر;\ اللَّ	charZZassociated_html شاركumpfassociated personaleucksにて 일을 일을 початку connectedازد달 erklкетemtepad Lista indem Eddemaisepad الأشياءعين略umpf الأشياءिकोepad beams :+: թույլ Ultra dessinée Ultra dartVII erkl ideिको filaments	close Spieltagкетpoly 일을_htmlumpf indem NT indem khaiिकोLFassociated指出 Perl_html personaleZZ ide德华 початкуिको	charumpf erklѓаにて compagnies-luv'assطوي Bras شارك expect початку filaments zwarountryZZланд personale початку شارك والمعduction Perlemais :+: 일을ланд Perl :+:ductionVII vuelo dessinée_html filamentsumpf
[danbev] Stopping generation due to parse error.
```
This model is a reasoning/thinking model and this is the default when using
llama-cli. So my theory is that the peg parser is created and looks at the
enable_thinking value and creates a grammar for this. If the output from model
does not meet this expected structure, and the model in question has random
weights so it probably will not, then the peg-parser might error? 
We can start llama-cli using `--reasoning-budget 0` to disable thinking and see
if the error persists.

In common_chat_params_init_nemotron_v3 we have:
```console
static common_chat_params common_chat_params_init_nemotron_v3(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_constructed_parser([&](auto & p) {
        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.reasoning(p.until("</think>")) + ("</think>" | p.end());
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            }
        }
```
So disabling thinking still causes an error to be thrown:
```console
[danbev] Parser error during generation]
[danbev] Exception: Failed to parse input at pos 0
[danbev] Content generated so far:
emais :+:ZZ erkl-luvкет mastersѓа_html federalump
```
But notice that it is complaining about pos 0 now instead of pos 1.

If we inspect the meta and chat_params objects in gdb we get:
```
(gdb) p meta
$1 = {build_info = "b7866-5994dfce9", model_name = "output-q4_k_m.gguf", model_path = "models/output-q4_k_m.gguf", has_mtmd = false, 
  has_inp_image = false, has_inp_audio = false, 
  json_webui_settings = {<nlohmann::json_abi_v3_12_0::detail::json_default_base> = {<No data fields>}, m_data = {
      m_type = nlohmann::json_abi_v3_12_0::detail::value_t::object, m_value = {object = 0xaaaab781d400, array = 0xaaaab781d400, 
        string = 0xaaaab781d400, binary = 0xaaaab781d400, boolean = false, number_integer = 187650199901184, 
        number_unsigned = 187650199901184, number_float = 9.2711517206417491e-310}}}, slot_n_ctx = 4096, 
  pooling_type = LLAMA_POOLING_TYPE_NONE, chat_params = @0xaaaaac46b890, chat_template_caps = std::map with 6 elements = {
    ["requires_typed_content"] = false, ["supports_parallel_tool_calls"] = true, ["supports_preserve_reasoning"] = true, 
    ["supports_system_role"] = true, ["supports_tool_calls"] = true, ["supports_tools"] = true}, bos_token_str = "<s>", 
  eos_token_str = "<|im_end|>", fim_pre_token = -1, fim_sub_token = -1, fim_mid_token = -1, model_vocab_type = LLAMA_VOCAB_TYPE_BPE, 
  model_vocab_n_tokens = 131072, model_n_ctx_train = 1048576, model_n_embd_inp = 4096, model_n_params = 120668707840, 
  model_size = 86173184000}
```
And if we look at the parser we can see:
```console
(gdb) p chat_params
$5 = {format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED, 
  prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n</think>", 
  grammar = "", grammar_lazy = false, thinking_forced_open = false, grammar_triggers = std::vector of length 0, capacity 0, 
  preserved_tokens = std::vector of length 4, capacity 4 = {"<think>", "</think>", "<tool_call>", "</tool_call>"}, 
  additional_stops = std::vector of length 0, capacity 0, 
  parser = "{\"parsers\":[{\"type\":\"epsilon\"},{\"delimiters\":[],\"type\":\"until\"},{\"child\":1,\"tag\":\"content\",\"type\":\"tag\"},{\"type\":\"space\"},{\"children\":[0,3,2],\"type\":\"sequence\"}],\"root\":4,\"rules\":{}}"}
```
```c++
        // Content only parser
        include_grammar = false;
        return reasoning << p.content(p.rest());
```


#### Conclusion
Adding some logging to [peg-parser.cpp](https://github.com/ggml-org/llama.cpp/blob/41ea26144e55d23f37bb765f88c07588d786567f/common/peg-parser.cpp#L637) produces:
```c++
if (utf8_result.status == utf8_parse_result::INVALID) {
 // Malformed UTF-8
 printf("\n[danbev] Invalid UTF-8 at pos %zu, start_pos: %zu\n", pos, start_pos);
 return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
}
```
Produces the following:
```console
> Hello

kitchen filaments Refresh الأشياء 일을 Frank naw姐 personale-luv	charWEB指出ланд_html personale vuelo filamentsिकोVII початкуemais kebumpf.Abstract}}-\ nazýemaisepad federalѓа jobb personale могут jobb erkl filamentsduction filaments doanucksازد্ম personale початку jobb entering compagnies崎 일을 Frankemais strawкетemais )),
 jobb z filaments 일을 filamentsкетEasyланд violin kitchen_htmlिकोланд 천 erkl personale شاركкет달 المباشر erkl personale달静 séries指出指出 erklductionassociated dessinée	char :+: الأشياء Perlкетemais شاركازد ideкет jobb personaleepad éte 천 شارك 천 filamentsductionattaque filaments 일을ازدिकोumpfumpfкетpoly erklिको شاركemtemais escalaucks الأشياء poneremais kitchen kontin崎emt الأشياءкетույթիкет近年来 могутازدкетZZѓа zwar্ম issus	char-luv指出 Kammer federalاصمة شارك beamsкеткет personale德华 jobb.contсій崎िकोductionumpfкет початку lecteurductionumpfѓаازدduction straw federal personale NT崎िकोumpfumpfVII federal erklкет.Abstract الأشياء-luv початкуumpfduction 일을্ম Perl thesis filaments_html початкуemtemaisкет شارك静 compagnies指出emaisumpf filaments욱 Ultra;\指出 Kammer filamentsstil.contкетugasassociated شاركemt الأشياء compagniesEasyланд zwar指出 federalumpf 일을 personaleemaisško-luv Ultraemais Spieltag jobb指出 الأشياء شاركemaisEasyassociated beamsumpfѓа NT Brasemais崎stil Ellesumpf 천 erkl unfavorableassociated 일을 lecteur therstilZZ erkl.cont початкукет 일을 mana beams-luv beamsumpfumpfкет compagniesemais달 personale filamentszailea宝кетѓа streng beams doan্মemtعينिको	char Bras달emais.contازد	char-luv personaleാറ്റ 일을.Abstract崎кет filaments Frank թույլ Migrationassociatedduction compagniesкет 일을 khai Ultra}}(ductionassociated federal崎 naw الأشياءѓаumpf姐ZZZZ指出 Lista.Abstract Jorgeumpf宝 STAR 천 початкуductionemais issus 일을指出 manaepad-luv початкукет Ultra federal指出 الأشياء指出.Abstract	charZZ姐.cont指出 dart թույլ-luv Confederate jobb federalipingemt pratiqueкетZZ artific erklिको崎associated erkl德华 UltraZZिकोVII Frank indem filaments 일을кетازد_htmlइ manaumpf_html erkl compagniesWEB Navbarpolyѓаemt పైstil달epad指出 Refresh崎 kitchen Frank erkl dart.Abstractعين interactingVII mana आज_html Perl zwar Ultra Spieltag strawкет달 keb Frankланд singing ide달 filaments Ultra personale dartにて.Abstractिको.Abstract指出ucksemais	char指出 personale :+:кет zwaremais-luv الأشياءिकोemaisucksкет filaments strawucks 셋_html dessinée্ম 일을武_html початку شارك崎指出Mahon Ultraѓа indemкетिकोountryemaisにて指出umpfasonumpfാറ്റ federal ide erkl Navbarduction filaments 일을ѓа Frank الأشياءланд audiences unfavorable పై erkl指出ZZ指出WEB compagniesumpfduction Refreshкет violin Frank德华umpfumpf compagnies beamsemais erkl姐 federal崎にて.cont )),
 :+:ѓа personale ide
[danbev] Invalid UTF-8 at pos 3352, start_pos: 1
```
So the model is producing invalid UTF-8 sequences which is likely causing the
peg-parser to fail. This would likely not happen with a trained modle and the
reason we are seeing this is because the model is using random weights.

And the caught exception for this:
```console
[danbev] Parser error during generation]
[danbev] Exception: Failed to parse input at pos 1
[danbev] Content generated so far:
kitchen filaments Refresh الأشياء 일을 Frank naw姐 personale-luv	charWEB指出ланд_html personale vuelo filamentsिकोVII початкуemais kebumpf.Abstract}}-\ nazýemaisepad federalѓа jobb personale могут jobb erkl filamentsduction filaments doanucksازد্ম personale початку jobb entering compagnies崎 일을 Frankemais strawкетemais )),
 jobb z filaments 일을 filamentsкетEasyланд violin kitchen_htmlिकोланд 천 erkl personale شاركкет달 المباشر erkl personale달静 séries指出指出 erklductionassociated dessinée	char :+: الأشياء Perlкетemais شاركازد ideкет jobb personaleepad éte 천 شارك 천 filamentsductionattaque filaments 일을ازدिकोumpfumpfкетpoly erklिको شاركemtemais escalaucks الأشياء poneremais kitchen kontin崎emt الأشياءкетույթիкет近年来 могутازدкетZZѓа zwar্ম issus	char-luv指出 Kammer federalاصمة شارك beamsкеткет personale德华 jobb.contсій崎िकोductionumpfкет початку lecteurductionumpfѓаازدduction straw federal personale NT崎िकोumpfumpfVII federal erklкет.Abstract الأشياء-luv початкуumpfduction 일을্ম Perl thesis filaments_html початкуemtemaisкет شارك静 compagnies指出emaisumpf filaments욱 Ultra;\指出 Kammer filamentsstil.contкетugasassociated شاركemt الأشياء compagniesEasyланд zwar指出 federalumpf 일을 personaleemaisško-luv Ultraemais Spieltag jobb指出 الأشياء شاركemaisEasyassociated beamsumpfѓа NT Brasemais崎stil Ellesumpf 천 erkl unfavorableassociated 일을 lecteur therstilZZ erkl.cont початкукет 일을 mana beams-luv beamsumpfumpfкет compagniesemais달 personale filamentszailea宝кетѓа streng beams doan্মemtعينिको	char Bras달emais.contازد	char-luv personaleാറ്റ 일을.Abstract崎кет filaments Frank թույլ Migrationassociatedduction compagniesкет 일을 khai Ultra}}(ductionassociated federal崎 naw الأشياءѓаumpf姐ZZZZ指出 Lista.Abstract Jorgeumpf宝 STAR 천 початкуductionemais issus 일을指出 manaepad-luv початкукет Ultra federal指出 الأشياء指出.Abstract	charZZ姐.cont指出 dart թույլ-luv Confederate jobb federalipingemt pratiqueкетZZ artific erklिको崎associated erkl德华 UltraZZिकोVII Frank indem filaments 일을кетازد_htmlइ manaumpf_html erkl compagniesWEB Navbarpolyѓаemt పైstil달epad指出 Refresh崎 kitchen Frank erkl dart.Abstractعين interactingVII mana आज_html Perl zwar Ultra Spieltag strawкет달 keb Frankланд singing ide달 filaments Ultra personale dartにて.Abstractिको.Abstract指出ucksemais	char指出 personale :+:кет zwaremais-luv الأشياءिकोemaisucksкет filaments strawucks 셋_html dessinée্ম 일을武_html початку شارك崎指出Mahon Ultraѓа indemкетिकोountryemaisにて指出umpfasonumpfാറ്റ federal ide erkl Navbarduction filaments 일을ѓа Frank الأشياءланд audiences unfavorable పై erkl指出ZZ指出WEB compagniesumpfduction Refreshкет violin Frank德华umpfumpf compagnies beamsemais erkl姐 federal崎にて.cont )),
 :+:ѓа personale ide
[danbev] Stopping generation due to parse error.
```
Since this is only reporting the start position (pos 1 above) this was a bit
confusing at first. Perhaps we could add a debug logging line to peg-parser.cpp to
clarify this.

To run a model with random weighs we can use `-no-jinja` which does not use the
peg-parser.
