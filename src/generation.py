import torch
import torch.nn.functional as F
import numpy as np

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping, tokenizer=None):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.length_fact = []
        self.beams = []
        self.worst_score = 1e9
        self.raw_worst_score = 1e9

        self.tokenizer = tokenizer

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # print(f'add hyp = {self.tokenizer.decode(hyp.cpu().tolist())}, score = {score}')
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            self.length_fact.append(len(hyp) ** self.length_penalty)
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx, _) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
                self.raw_worst_score = self.worst_score * (len(sorted_scores[1][2]) ** self.length_penalty)
            else:
                self.worst_score = min(score, self.worst_score)
                self.raw_worst_score = sum_logprobs
        
        # print('maintained hypothesis: ')
        # for score, hyp in self.beams:
        #     print(f'raw_score = {score * (len(hyp) ** self.length_penalty)}, score = {score}, hyp = {self.tokenizer.decode(hyp.cpu().tolist())}')

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # print(f'cur best score = {cur_score}, cur worst score = {self.worst_score}, cur raw worst score = {self.raw_worst_score}')
            ret = self.worst_score >= cur_score

            # print("in beam")
            # for x in self.beams:
            #     print(x[0], self.tokenizer.decode(x[1].cpu().tolist()))
            # print("end beam")

            return ret


def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, tokenizer):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    # cur_len = prev_input_ids.size(-1)
    # # prev_input_words = tokenizer.decode(prev)
    # if cur_len + 1 < no_repeat_ngram_size:
    #     # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
    #     return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    prev_input_words = []
    for ids in prev_input_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
        words = []
        for token in tokens:
            token = token.replace("</_>", "") # NOTE: "▁" is different from "_"
            if len(token) > 0:
                if token in ['<sep>', "<unk>", "<s>", "</s>", "<eod>", "<mask>"]:
                    words.append(token)
                else:
                    words += list(token)
        prev_input_words.append(words)
    # print(prev_input_words)
    for idx in range(num_hypos):
        gen_words = prev_input_words[idx]
        # print('gen_words = ', gen_words)
        # gen_tokens = prev_input_ids[idx].tolist()
        # gen_words = tokenizer.decode(gen_tokens)
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_words[i:] for i in range(no_repeat_ngram_size)]):
            for prefix_len in range(no_repeat_ngram_size):
                prev_ngram = ''.join(ngram[:prefix_len])
                if "</n>" not in prev_ngram:
                    suffix_ngram = ''.join(ngram[prefix_len:])
                    suffix_ngram_2 = "▁" + suffix_ngram
                    if tokenizer.check(suffix_ngram): # 在词表中
                        generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram])
                    if tokenizer.check(suffix_ngram_2): # 在词表中
                        generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram_2])
            # prev_ngram_tuple = ''.join(ngram[:-1])
            # generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, set()) | set([ngram[-1]])
    # for g in generated_ngrams:
    #     print(g)
    # print('generated_ngrams = ', generated_ngrams)

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared

        cur_len = len(prev_input_words[hypo_idx])
        
        generated_ngram_idx = []
        for prefix_len in range(no_repeat_ngram_size):
            # print('')
            ngram_words = ''.join(prev_input_words[hypo_idx][cur_len-prefix_len:])
            # print('prev_input = ', prev_input_words[hypo_idx])
            # print('ngram_words = ', ngram_words)
            generated_ngram_words = generated_ngrams[hypo_idx].get(ngram_words, [])
            # print('generated_ngram_words = ', generated_ngram_words)
            # print('all generated_ngrams = ', generated_ngrams[hypo_idx])
            generated_ngram_idx += tokenizer.convert_tokens_to_ids(generated_ngram_words)
            # generated_ngram_idx += [x for word in generated_ngram_words for x in tokenizer.get_prefix_id_list(word)]
            # print('generated_ngram_idx = ', generated_ngram_idx)
            # print('='*100)
        prev_input_str = "".join(prev_input_words[hypo_idx])
        # print("prev input str", prev_input_str)
        if prev_input_str[-1] in ['，', ',']:
            generated_ngram_idx.append(tokenizer.convert_token_to_id('但'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('▁但'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id(','))
        if prev_input_str[-2:] in ["我是", "我叫"]:
            generated_ngram_idx.append(tokenizer.convert_token_to_id(','))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('▁.'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('.'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('。'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('▃'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('—'))


        return generated_ngram_idx

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-10000):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1), descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def enforce_repetition_penalty_(tokenizer, 
                                lprobs, 
                                batch_size, 
                                num_beams, 
                                prev_output_tokens, 
                                repetition_penalty):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            if previous_token != tokenizer.sep_id:
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(tokenizer,
                                  scores,
                                  input_ids,
                                  no_repeat_ngram_size,
                                  bad_words_ids,
                                  repetition_penalty,
                                  batch_size,
                                  num_beams):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            tokenizer, scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    # if eos_token_id is not None and cur_len < min_length:
    #     scores[:, eos_token_id] = -10000

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, tokenizer=tokenizer)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -10000

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -10000

    scores[:, 0] = -50000
    scores[:, 1] = -50000
    scores[:, 2] = -50000
    scores[:, 3] = -50000
    scores[:, 4] = -50000
    scores[:, 5] = -50000
    scores[:, 6] = -50000
    scores[:, 7] = -50000
    scores[:, 8] = -50000

    return scores


def round_up(x, d):
    return (x + d - 1) // d * d

def make_input(lef_tokens, rig_tokens, spans):
    input = lef_tokens + [0 for i in range(spans)] + rig_tokens
    length = len(input)

    rounded_length = round_up(length, 4)

    input_tokens = torch.zeros(1, rounded_length, dtype=torch.int32)
    input_span = torch.zeros(1, rounded_length, dtype=torch.int32)
    
    context = np.arange((rounded_length))
    context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
    context = torch.from_numpy(context).view(1, -1).bool()

    input_length = torch.zeros(1, dtype=torch.int32)
    input_tokens[0, :length] = torch.tensor(input).int()
    input_length[0] = length

    return input_tokens.cuda(), input_length.cuda(), input_span.cuda(), context.cuda()


def generate_no_beam(model, tokenizer, lef_sentence, rig_sentence, spans, 
                     temperature = .9, top_k = 0, top_p = 0.9,
                     no_repeat_ngram_size = 3, repetition_penalty = 1):

    lef_tokens = tokenizer.encode(lef_sentence)
    rig_tokens = tokenizer.encode(rig_sentence)
    lef_tokens = [1] + lef_tokens

    input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)

    for i in range(len(lef_tokens) - 1, len(lef_tokens) + spans - 1):
        logits = model(input_tokens, input_length, context, input_span)
        logits = logits[:, i, :] / temperature
        logits = postprocess_next_token_scores(
            tokenizer=tokenizer,
            scores=logits,
            input_ids=input_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=[[0]],
            repetition_penalty=repetition_penalty,
            batch_size=1,
            num_beams=1,
        )
        logits = top_k_logits(logits, top_k = top_k, top_p = top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_tokens[0][i + 1] = next_token
        # context[0][i+1] = True
    for i in input_tokens[0].cpu().numpy():
        yield tokenizer.decode([i])


# def generate_beam(model_batch, token_tensor_full, model, tokenizer, args, device):
#     batch_size = args.batch_size
#     num_beams = args.num_beams
#     target_length = args.max_length
    
#     do_sample = args.top_p > 0 or args.top_k > 0
#     vocab_size = tokenizer.vocab_size
    
#     input_ids = model_batch['input_ids']
#     attention_mask = model_batch['attention_mask']
#     position_ids = model_batch['position_ids']
#     # we use past_key_values, so only the current token mask is needed
#     init_length = input_ids.size(-1)
    
#     input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, init_length)
#     attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, 1, init_length, init_length)
#     position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, init_length)
#     token_tensor_full = token_tensor_full.unsqueeze(1).expand(batch_size, num_beams, token_tensor_full.size(-1))

#     input_ids = input_ids.contiguous().view(batch_size * num_beams, init_length)
#     attention_mask = attention_mask.contiguous().view(batch_size * num_beams, 1, init_length, init_length)
#     position_ids = position_ids.contiguous().view(batch_size * num_beams, init_length)
#     token_tensor_full = token_tensor_full.contiguous().view(batch_size * num_beams, token_tensor_full.size(-1))

#     done = [False for _ in range(batch_size)]
#     output_ids = input_ids.new_zeros([input_ids.size(0), 0]) # not include the prompt
#     past_key_values = None
    
#     gen_len = 0

#     # generated hypotheses
#     generated_hyps = [
#         BeamHypotheses(num_beams, target_length, args.length_penalty, early_stopping=args.early_stopping, tokenizer=tokenizer)
#         for _ in range(batch_size)
#     ]

#     beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
#     beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

#     # print(tokenizer.decode(input_ids[0].cpu().tolist()).replace("\n", "<n>"))
#     # print(position_ids)

#     while gen_len < target_length:
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#         )
#         past_key_values = outputs['past_key_values']

#         lm_logits = outputs["lm_logits"]

#         logits = lm_logits[:, -1, :] / args.temperature
#         scores = F.log_softmax(logits, dim=-1)

#         prev_output_tokens = torch.cat([token_tensor_full, output_ids], dim=-1)

#         scores = postprocess_next_token_scores(
#             tokenizer=tokenizer,
#             scores=scores,
#             input_ids=prev_output_tokens,
#             no_repeat_ngram_size=args.no_repeat_ngram_size,
#             bad_words_ids=None,
#             repetition_penalty=args.repetition_penalty,
#             batch_size=batch_size,
#             num_beams=num_beams,
#         )
#         if do_sample:
#             _scores = scores + beam_scores[:, None].expand_as(scores)
#             if args.temperature != 1.0:
#                 _scores = _scores / args.temperature                
#             _scores = top_k_logits(_scores, top_k=args.top_k, top_p=args.top_p)
#             _scores = _scores.contiguous().view(batch_size, num_beams * vocab_size)
#             # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
#             probs = F.softmax(_scores, dim=-1)
#             next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
#             # Compute next scores
#             next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
#             # sort the sampled vector to make sure that the first num_beams samples are the best
#             next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
#             next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)            
#         else:
#             next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

#             # re-organize to group the beam together (we are keeping top hypothesis accross beams)
#             next_scores = next_scores.view(
#                 batch_size, num_beams * vocab_size
#             )  # (batch_size, num_beams * vocab_size)

#             next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

#         assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
#         # next batch beam content
#         next_batch_beam = []

#         for batch_idx in range(batch_size):
#             # if we are done with this sentence, add a pad token
#             if done[batch_idx]:
#                 assert (
#                     len(generated_hyps[batch_idx]) >= num_beams
#                 ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
#                 next_batch_beam.extend([(0, tokenizer.pad_id, 0)] * num_beams)  # pad the batch
#                 continue

#             # next sentence beam content, this will get added to next_batch_beam
#             next_sent_beam = []

#             # next tokens for this sentence
#             for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
#                 zip(next_tokens[batch_idx], next_scores[batch_idx])
#             ):
#                 # get beam and token IDs
#                 beam_id = beam_token_id // vocab_size
#                 token_id = beam_token_id % vocab_size

#                 effective_beam_id = batch_idx * num_beams + beam_id
#                 token_set = {}
#                 for t in output_ids[effective_beam_id]:
#                     token_set[t] = token_set[t] + 1 if t in token_set else 1
#                 # add to generated hypotheses if end of sentence
#                 if token_id.item() == 3 or sum([token_set.get(t, 0) for t in [12, 36, 45, 29594]]) > 0:
#                     # if beam_token does not belong to top num_beams tokens, it should not be added
#                     is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
#                     if is_beam_token_worse_than_top_num_beams:
#                         continue
#                     generated_hyps[batch_idx].add(
#                         output_ids[effective_beam_id].clone(), beam_token_score.item(),
#                     )
#                 else:
#                     # add next predicted token since it is not eos_token
#                     next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

#                 # once the beam for next step is full, don't add more tokens to it.
#                 if len(next_sent_beam) == num_beams:
#                     break

#             # Check if we are done so that we can save a pad step if all(done)
#             # is_done: the best candiates in the current beam is worse than the sentences already in generated_hyps
#             # print('cur worst score = ', generated_hyps[batch_idx].worst_score)
#             # print('cur raw worst score = ', generated_hyps[batch_idx].raw_worst_score)
#             done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
#                 next_scores[batch_idx].max().item(), gen_len
#             )
#             # for score, token_id, effective_beam_id in next_sent_beam:
#             #     print(f'raw_socre = {score}, score = {score / gen_len ** args.length_penalty}, sentence = {tokenizer.decode(torch.cat([output_ids[effective_beam_id], token_id.unsqueeze(dim=0)], dim=-1).cpu().tolist())}')
#             # print(f'id_done = {done[batch_idx]}')
#             # print('='*100)

#             # update next beam content
#             assert len(next_sent_beam) == num_beams, "Beam should always be full"
#             next_batch_beam.extend(next_sent_beam)
#             assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

#         # stop when we are done with each sentence
#         if all(done):
#             break

#         # sanity check / prepare next batch
#         assert len(next_batch_beam) == batch_size * num_beams
#         beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=input_ids.device)
#         beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=input_ids.device)
#         beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=input_ids.device)

#         # re-order batch and update current length
#         output_ids = output_ids[beam_idx, :]
#         output_ids = torch.cat([output_ids, beam_tokens.unsqueeze(1)], dim=-1)

#         # print(beam_scores)
#         # for ol in output_ids.cpu().tolist():
#         #     print(tokenizer.decode(ol))

#         input_ids = beam_tokens.unsqueeze(1)
#         attention_mask = torch.cat([attention_mask[:, :, -1:, :], attention_mask[:, :, -1:, -1:]], dim=-1)
#         position_ids = position_ids[:, -1:] + 1
#         past_key_values = [[torch.index_select(layer_past_type, 0, beam_idx) for layer_past_type in layer_past] for layer_past in past_key_values]
#         gen_len += 1

#     # finalize all open beam hypotheses and add to generated hypotheses
#     for batch_idx in range(batch_size):
#         if done[batch_idx]:
#             continue

#         # need to add best num_beams hypotheses to generated hyps
#         for beam_id in range(num_beams):
#             effective_beam_id = batch_idx * num_beams + beam_id
#             final_score = beam_scores[effective_beam_id].item()
#             final_tokens = output_ids[effective_beam_id]
#             generated_hyps[batch_idx].add(final_tokens, final_score)

#     best = []
#     best_ids = []

#     # retrieve best hypotheses
#     for i, hypotheses in enumerate(generated_hyps):
#         sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
#         # for score, hyp in sorted_hyps:
#         #     print(f'score = {score}, hyp = {tokenizer.decode(hyp.cpu().tolist())}')
#         best_hyp = sorted_hyps.pop()[1]
#         if best_hyp[-1] == 8:
#             best_hyp = best_hyp[:-1]
#         best.append(tokenizer.decode(best_hyp.cpu().tolist()))
#         best_ids.append(best_hyp.cpu().tolist())

#     return best, best_ids


# def generate(model_batch, token_tensor_full, model, tokenizer, args, device):
#     if args.num_beams == 1:
#         generation_str_list, generation_id_list = generate_no_beam(model_batch, token_tensor_full, model, tokenizer, args, device)
#     else:
#         generation_str_list, generation_id_list = generate_beam(model_batch, token_tensor_full, model, tokenizer, args, device)

#     return generation_str_list, generation_id_list
