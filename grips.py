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
import torch
nltk.download('punkt')

parser = Parser.load('crf-con-en')


para_model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(
    torch_device).eval()

level = 'phrase'

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
        print(f"tree {tree}")
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
        print(f"parse {sentence}")
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=True).sentences[0].trees[0]
        print(f"parsed_tree {parsed_tree}")
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    print(f"phrases: {phrases}")
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if
                phrase not in string.punctuation or phrase == '']
    print(f"phrases: {phrases}")
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


def grips(
        instruction,
        score_fn,
        num_steps=10,
        edit_operations_ = ('del', 'swap', 'sub', 'add'),
        simulated_anneal = False,
        num_compose = 1,
        num_candidates = 5,
        patience = 2,
):
    edit_operations = list(edit_operations_)
    use_add = 'add' in edit_operations

    operations_tracker = []
    base_candidate = detokenize(word_tokenize(instruction))
    assert word_tokenize(base_candidate) == word_tokenize(instruction)
    original_candidate = base_candidate
    base_score = score_fn([base_candidate])[0]
    orig_score = base_score
    print(f"base candidate score: {base_score}")
    delete_tracker = []
    patience_counter = 1
    for i in range(num_steps):
        print(f"step {i}")
        deleted = {}
        added = {}
        print("phrase_lookup")
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
            print(f"edit {edit}")
            if isinstance(edit, str):
                # print(f'perform_edit {edit}')
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                # print(f'candidate {candidate}')
                candidates.append(candidate)
                if edit == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == 'add':
                    if len(indices): added[candidate] = indices
            else:
                old_candidate = base_candidate
                composed_deletes = []
                composed_adds = []
                # print(f'compsed {edit}')
                for op in edit:
                    phrase_lookup = get_phrase_lookup(old_candidate)
                    new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup,
                                                            delete_tracker)
                    if op == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                    if op == 'add':
                        if len(indices): composed_adds.append(indices[0])
                    old_candidate = new_candidate
                # print(f'candidate {new_candidate}')
                candidates.append(new_candidate)
                if 'del' in edit: deleted[new_candidate] = composed_deletes
                if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds

        print(f'candidates {candidates}')
        print(base_score)
        scores = score_fn(candidates)
        print(scores)
        print(f'beaten base candidate: {base_score > orig_score}')
        # for c, candidate in enumerate(candidates):
        #     scores.append(score(candidate))
        #     print(scores[-1])

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
                T_max = 10
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

    return base_candidate