import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import argparse
# from nat_inst_gpt3 import *
from sklearn.metrics import balanced_accuracy_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import entropy
import json
nltk.download('punkt')



PROMPT_FORMAT = """{prompt}
Input: {x}
Output: {y}"""

def grips(
        instruction,
        mode,
        examples, task_labels,
        batch_size, num_shots,
        num_compose, num_candidates, num_steps,
        num_samples = 100,
        prompt_format=PROMPT_FORMAT,
        train_seed=423523,
        edits=('del', 'swap', 'sub', 'add'),
        meta_name='meta',
        meta_dir='.',
        level='phrase',
        simulated_anneal=False,
        patience=2
):
    seed = train_seed
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    task_labels = list(set(task_labels))
    task_labels.sort()
    print(task_labels)

    num_train_samples = len(examples)

    chosen_task_name = None

    # instruction = file_contents['Definition']
    instruction = instruction.replace('\n' + 'Things to avoid: -', '')
    instruction = instruction.replace('\n' + 'Emphasis & Caution: -', '')
    parser = Parser.load('crf-con-en')
    T_max = 10
    edit_operations = edits
    use_add = 'add' in edit_operations

    if 'sub' in edit_operations:
        para_model_name = 'tuner007/pegasus_paraphrase'
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
        para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(
            torch_device).eval()


    def detokenize(tokens):
        return TreebankWordDetokenizer().detokenize(tokens)


    def traverse_tree(parsed_tree):
        phrases = []
        for tree in parsed_tree:
            if tree.label() == '_': continue
            phrases.append(detokenize(tree.leaves()))
            for subtree in tree:
                if type(subtree) == nltk.tree.Tree:
                    if subtree.label() == '_': continue
                    phrases.append(detokenize(subtree.leaves()))
                    phrases.extend(traverse_tree(subtree))
        return phrases


    def check_child(tree):
        check = False
        count = 0
        total_count = 0
        for subtree in tree:
            total_count += 1
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_':
                    count += 1
        if count >= total_count - count: check = True

        return check


    def collect_leaves(parsed_tree):
        leaves = []
        for tree in parsed_tree:
            if type(parsed_tree) != nltk.tree.Tree: continue
            if tree.label() == '_':
                leaves.append(detokenize(tree.leaves()))
                continue
            if check_child(tree):
                leaves.append(detokenize(tree.leaves()))
            else:
                leaves.extend(collect_leaves(tree))
        return leaves


    def get_phrases(instruction):  # one possible way of obtaining disjoint phrases
        phrases = []
        for sentence in sent_tokenize(instruction):
            parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
            leaves = collect_leaves(parsed_tree)
            phrases.extend(leaves)
        phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if
                   phrase not in string.punctuation or phrase == '']
        return phrases


    def get_response(input_text, num_return_sequences, num_beams):
        batch = para_tokenizer([input_text], truncation=True, padding='longest', max_length=60,
                               return_tensors="pt").to(torch_device)
        translated = para_model.generate(**batch, max_length=60, num_beams=num_beams,
                                         num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text


    def delete_phrase(candidate, phrase):
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ')
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', ' ')
        else:
            answer = candidate.replace(phrase, '')
        return answer


    def add_phrase(candidate, phrase, after):
        if after == '':
            answer = phrase + ' ' + candidate
        else:
            if candidate.find(' ' + after) > 0:
                answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
            elif candidate.find(after + ' ') > 0:
                answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
            else:
                answer = candidate.replace(after, after + phrase)
        return answer


    def swap_phrases(candidate, phrase_1, phrase_2):
        if candidate.find(' ' + phrase_1 + ' ') >= 0:
            answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
        else:
            answer = candidate.replace(phrase_1, '<1>')
        if candidate.find(' ' + phrase_2 + ' ') >= 0:
            answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
        else:
            answer = candidate.replace(phrase_2, '<2>')
        answer = answer.replace('<1>', phrase_2)
        answer = answer.replace('<2>', phrase_1)
        return answer


    def substitute_phrase(candidate, phrase):
        num_beams = 10
        num_return_sequences = 10
        paraphrases = get_response(phrase, num_return_sequences, num_beams)
        paraphrase = np.random.choice(paraphrases, 1)[0]
        paraphrase = paraphrase.strip('.')
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', paraphrase + ' ')
        else:
            answer = candidate.replace(phrase, paraphrase)
        return answer


    def perform_edit(edit, base, phrase_lookup, delete_tracker):
        if edit == 'del':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return delete_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'swap':
            try:
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False)
            except:
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True)
            return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
        elif edit == 'sub':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return substitute_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'add':
            keys = list(phrase_lookup.keys())
            keys.append(-1)
            [i] = np.random.choice(keys, 1)
            if i >= 0:
                after = phrase_lookup[i]
            else:
                after = ''
            if len(delete_tracker) == 0: return base, []
            phrase = np.random.choice(delete_tracker, 1)[0]
            return add_phrase(base, phrase, after), [phrase]

    # def rob_prompt(mode, task_name, num_shots, num_test_instances, seed, split, modified):
    #     return prompt_list, answer_list, index_list


    def custom_instruction_prompt(mode=mode, task_name=chosen_task_name, num_shots=num_shots,
                                  num_test_instances=num_samples, seed=seed, null_word=None,
                                  split='train', modified={}):
        if mode == "Instruction Only":
            prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encodeinstruction(
                task_name, instruction_structure=["Definition"], number_of_examples=num_shots,
                number_of_instances=num_test_instances, seed=seed, null_word=null_word,
                modified=modified)
        elif mode == "Instruction + Positive Examples":
            prompt_list, answer_list, index_list, train_prompt_list, train_answer_list, train_index_list, dev_prompt_list, dev_answer_list, dev_index_list = training_encodeinstruction(
                task_name, instruction_structure=["Definition", "Positive Examples Full Only"],
                number_of_examples=num_shots, number_of_instances=num_test_instances, seed=seed,
                null_word=null_word, modified=modified)
        else:
            raise ValueError()
        if split == 'test':
            return prompt_list, answer_list, index_list
        elif split == 'train':
            train_prompt_list.extend(dev_prompt_list)
            train_answer_list.extend(dev_answer_list)
            train_index_list.extend(dev_index_list)
            try:
                random.seed(seed)
                indices = random.sample(range(len(train_index_list)), num_train_samples)
                train_prompt_list = [train_prompt_list[i] for i in indices]
                train_answer_list = [train_answer_list[i] for i in indices]
                train_index_list = [train_index_list[i] for i in indices]
            except:
                pass

            return train_prompt_list, train_answer_list, train_index_list

        else:
            raise ValueError()


    def score(candidate, split='train', write=False):
        label_probs, calibrated_label_probs, raw_acc_count, raw_cal_acc_count, answer_list, index_list, _ = run(
            mode=mode, batch_size=batch_size, num_shots=num_shots, chosen_task_name=chosen_task_name,
            num_samples=num_samples, seed=seed, override_prompts=True,
            function=custom_instruction_prompt, split=split, modified={'Definition': candidate},
            task_labels=task_labels, if_calibrate=False)
        preds = get_prediction(label_probs, task_labels)
        raw_acc = balanced_accuracy_score(answer_list, preds)
        label_frequencies = [preds.count(l) / len(preds) for l in task_labels]
        if split == 'train':
            return np.round(100 * raw_acc, 2) + 10 * entropy(label_frequencies)
        elif split == 'test':
            if write:
                pname = meta_name
                pname = pname.split('.')[0] + "_predictions.json"
                pred_dump = {'predictions': preds, 'answers': answer_list, 'ids': index_list}
                ppath = os.path.join(meta_dir, pname)
                pfile = open(ppath, 'w+')
                json.dump(pred_dump, pfile)
            return np.round(100 * raw_acc_count / len(answer_list), 2)
        else:
            return


    def get_phrase_lookup(base_candidate):
        if level == 'phrase':
            phrase_lookup = {p: phrase for p, phrase in enumerate(get_phrases(base_candidate))}
        elif level == 'word':
            words = word_tokenize(base_candidate)
            words = [w for w in words if w not in string.punctuation or w != '']
            phrase_lookup = {p: phrase for p, phrase in enumerate(words)}
        elif level == 'sentence':
            sentences = sent_tokenize(base_candidate)
            phrase_lookup = {p: phrase for p, phrase in enumerate(sentences)}
        elif level == 'span':
            phrases = []
            for sentence in sent_tokenize(base_candidate):
                spans_per_sentence = np.random.choice(
                    range(2, 5))  # split sentence into 2, 3, 4, 5 chunks
                spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
                spans = [detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p: phrase for p, phrase in enumerate(phrases)}
        else:
            raise ValueError()
        return phrase_lookup


    operations_tracker = []
    base_candidate = detokenize(word_tokenize(instruction))
    assert word_tokenize(base_candidate) == word_tokenize(instruction)
    original_candidate = base_candidate
    base_score = score(base_candidate)
    delete_tracker = []
    patience_counter = 1
    for i in range(num_steps):
        deleted = {}
        added = {}
        phrase_lookup = get_phrase_lookup(base_candidate)
        if base_candidate == original_candidate:
            for p in phrase_lookup.values(): print(p)
        if use_add:
            if len(delete_tracker):
                if 'add' not in edit_operations: edit_operations.append('add')
            else:
                if 'add' in edit_operations: edit_operations.remove('add')
        if num_compose == 1:
            edits = np.random.choice(edit_operations, num_candidates)
        else:
            edits = []
            for n in range(num_candidates):
                edits.append(np.random.choice(edit_operations, num_compose))
        print(edits)

        # generate candidates
        candidates = []
        for edit in edits:
            if isinstance(edit, str):
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                candidates.append(candidate)
                if edit == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == 'add':
                    if len(indices): added[candidate] = indices
            else:
                old_candidate = base_candidate
                composed_deletes = []
                composed_adds = []
                for op in edit:
                    phrase_lookup = get_phrase_lookup(old_candidate)
                    new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup,
                                                          delete_tracker)
                    if op == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                    if op == 'add':
                        if len(indices): composed_adds.append(indices[0])
                    old_candidate = new_candidate
                candidates.append(new_candidate)
                if 'del' in edit: deleted[new_candidate] = composed_deletes
                if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds

        print(base_score)
        scores = []
        for c, candidate in enumerate(candidates):
            scores.append(score(candidate))
            print(scores[-1])

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        if best_score > base_score:
            patience_counter = 1
            base_candidate = candidates[best_idx]
            base_score = best_score
            operations_tracker.append(edits[best_idx])
            print('New Base Candidate: ', base_candidate)
            if base_candidate in added.keys():
                print('Notice! Prev tracker: ', delete_tracker)
                for chunk in added[base_candidate]:
                    try:
                        delete_tracker.remove(chunk)
                    except:
                        pass
                print('Notice! New tracker: ', delete_tracker)
            if base_candidate in deleted.keys():
                delete_tracker.extend(deleted[base_candidate])
            base_candidate = detokenize(word_tokenize(base_candidate))

        else:
            patience_counter += 1

            if simulated_anneal:
                K = 5
                T = T_max * np.exp(-i / K)
                idx = np.argmax(scores)
                chosen_score = scores[idx]
                prob = np.exp((chosen_score - base_score) / T)
                if np.random.binomial(1, prob):
                    print('\n')
                    print('Update from simulated anneal')
                    base_candidate = candidates[idx]
                    base_score = chosen_score
                    print('New Base Candidate: ' + base_candidate)
                    if base_candidate in added.keys():
                        print('Notice! Prev tracker: ', delete_tracker)
                        for chunk in added[base_candidate]:
                            try:
                                delete_tracker.remove(chunk)
                            except:
                                pass
                        print('Notice! New tracker: ', delete_tracker)
                    if base_candidate in deleted.keys():
                        delete_tracker.extend(deleted[base_candidate])
                    base_candidate = detokenize(word_tokenize(base_candidate))
                else:
                    if patience_counter > patience:
                        print('Ran out of patience')
                        break
                    else:
                        continue


            else:
                if patience_counter > patience:
                    print('Ran out of patience')
                    break
                else:
                    continue

    print('\nTesting .... ')
    if True:
        print('Task:\t', chosen_task_name)
        print('Original Instruction:\t', original_candidate)
        orig_score = score(original_candidate, 'test')
        print('Original Accuracy:\t', str(orig_score))

    if base_candidate == original_candidate:
        print('No viable candidate found!')
        exit()
    searched_score = score(base_candidate, 'test', write=False)
    print('Accuracy after search:\t', str(searched_score))
    print('Instruction after search:\t', base_candidate)
    print('Edit Operations:\t', operations_tracker)



def foo():
    operations_tracker = []
    base_candidate = detokenize(word_tokenize(instruction))
    assert word_tokenize(base_candidate) == word_tokenize(instruction)
    original_candidate = base_candidate
    base_score = score(base_candidate)
    delete_tracker = []
    patience_counter = 1
    for i in range(num_steps):
        deleted = {}
        added = {}
        phrase_lookup = get_phrase_lookup(base_candidate)
        if base_candidate == original_candidate:
            for p in phrase_lookup.values(): print(p)
        if use_add:
            if len(delete_tracker):
                if 'add' not in edit_operations: edit_operations.append('add')
            else:
                if 'add' in edit_operations: edit_operations.remove('add')
        if num_compose == 1:
            edits = np.random.choice(edit_operations, num_candidates)
        else:
            edits = []
            for n in range(num_candidates):
                edits.append(np.random.choice(edit_operations, num_compose))
        print(edits)

        # generate candidates
        candidates = []
        for edit in edits:
            if isinstance(edit, str):
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                candidates.append(candidate)
                if edit == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == 'add':
                    if len(indices): added[candidate] = indices
            else:
                old_candidate = base_candidate
                composed_deletes = []
                composed_adds = []
                for op in edit:
                    phrase_lookup = get_phrase_lookup(old_candidate)
                    new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup,
                                                          delete_tracker)
                    if op == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                    if op == 'add':
                        if len(indices): composed_adds.append(indices[0])
                    old_candidate = new_candidate
                candidates.append(new_candidate)
                if 'del' in edit: deleted[new_candidate] = composed_deletes
                if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds

        print(base_score)
        scores = []
        for c, candidate in enumerate(candidates):
            scores.append(score(candidate))
            print(scores[-1])

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        if best_score > base_score:
            patience_counter = 1
            base_candidate = candidates[best_idx]
            base_score = best_score
            operations_tracker.append(edits[best_idx])
            try:
            except:
            print('New Base Candidate: ', base_candidate)
            if base_candidate in added.keys():
                print('Notice! Prev tracker: ', delete_tracker)
                for chunk in added[base_candidate]:
                    try:
                        delete_tracker.remove(chunk)
                    except:
                        pass
                print('Notice! New tracker: ', delete_tracker)
            if base_candidate in deleted.keys():
                delete_tracker.extend(deleted[base_candidate])
            base_candidate = detokenize(word_tokenize(base_candidate))

        else:
            patience_counter += 1

            if simulated_anneal:
                K = 5
                T = T_max * np.exp(-i / K)
                idx = np.argmax(scores)
                chosen_score = scores[idx]
                prob = np.exp((chosen_score - base_score) / T)
                if np.random.binomial(1, prob):
                    print('\n')
                    print('Update from simulated anneal')
                    base_candidate = candidates[idx]
                    base_score = chosen_score
                    print('New Base Candidate: ' + base_candidate)
                    if base_candidate in added.keys():
                        print('Notice! Prev tracker: ', delete_tracker)
                        for chunk in added[base_candidate]:
                            try:
                                delete_tracker.remove(chunk)
                            except:
                                pass
                        print('Notice! New tracker: ', delete_tracker)
                    if base_candidate in deleted.keys():
                        delete_tracker.extend(deleted[base_candidate])
                    base_candidate = detokenize(word_tokenize(base_candidate))
                else:
                    if patience_counter > patience:
                        print('Ran out of patience')
                        break
                    else:
                        continue


            else:
                if patience_counter > patience:
                    print('Ran out of patience')
                    break
                else:
                    continue
