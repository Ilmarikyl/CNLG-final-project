import finmeter, random, lwvlib, json, re
from uralicNLP import semfi
from uralicNLP import uralicApi
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np
from numpy.random import choice
from markovchain import JsonStorage
from markovchain.text import MarkovText, ReplyMode


wv=lwvlib.load("fin-word2vec-lemma.bin", 10000, 500000)
vowels = ['a','e','i','o','u','y','ä','ö']
markov = MarkovText.from_file('kalevala_and_others_markov.json')
fallback_markov = MarkovText.from_file('fallback_markov.json')


def count_syllables(verse):
    ''' Count syllables in a verse '''

    tokenizer = RegexpTokenizer(r'\w+')
    n_syllables = 0
    for word in tokenizer.tokenize(verse):

        try:
            n_syllables += len(finmeter.hyphenate(word).split("-"))
        except Exception as e:
            pass
            # print(e)
            # print(verse)
            # print("Error täällä: count_syllables")

    return n_syllables


def most_frequent(List):
    ''' Return the most frequent element in a list '''

    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 


def tokenize_and_lemmatize(words):
    ''' Tokenize and lemmatize input. Returns a list of lemmas. '''

    tokenizer = RegexpTokenizer(r'\w+')
    lemmas = []

    for word in tokenizer.tokenize(words):
        for i, analysis in enumerate(uralicApi.analyze(word.lower(), "fin")):
            if analysis[0].split('+', 1)[1] == 'N+Sg+Nom':
                lemmas.append(analysis[0].split('+')[0])

    if len(lemmas) < 2:
        print("Try with some other nouns.")
        exit()

    return lemmas


def has_monosyllabic_word(verse):
    ''' Check if verse has a monosyllabic word '''

    verse = remove_extra_material(verse)
    for word in verse.split():
        stripped = word.strip(",.:")
        if count_syllables(stripped) == 1 or len(stripped) == 1:
            return True

    return False


def get_pos_template(lemmas):
    ''' Return the POSses of each input word. E.g. "NN". Currently only searches for NN. '''

    pos_template = ""
    for lemma in lemmas:
        candidates = []
        for analysis in uralicApi.analyze(lemma, "fin"):
            pos = analysis[0].split('+')[1]
            # print(analysis)
            if pos == 'N':
                candidates.append(pos)
        pos_template += most_frequent(candidates)

    return pos_template

def contains_back_vowels(word):
    ''' Checks if word contains back vowels '''

    for letter in word:
        if letter in ['a','o','u']:
            return True

    return False

def remove_extra_material(verse):
    ''' Removes some redundant characters '''

    to_remove_regex = re.compile('[0-9.,]')
    to_replace_regex = re.compile('["*:%–”_\[\]\(\)-]')
    verse = to_remove_regex.sub('', verse)
    verse = to_replace_regex.sub(' ', verse)
    verse = re.sub(' +', ' ', verse)

    return verse


def add_two_syllables_in_front(verse):
    verse = remove_extra_material(verse)
    try:
        if len(verse.split()) > 2 and verse.split()[0][1] in vowels:
            prefix_thing = verse[:2] + random.choice(['k','t','p','s']) + uralicApi.analyze(verse.split(" ")[0], "fin")[0][0].split('+')[0][-1] + " "
            verse = (prefix_thing + verse).capitalize()
        else:
            return verse
    except:
        pass

    return verse


def fix_syllables(verse, markov_seq=True):
    ''' This function should do something when the amount of syllables is not correct. It is not properly implemented yet. '''

    if count_syllables(verse) == 5:
        for word in verse.split():
            word = remove_extra_material(word)
            if word[-2:] in ['aa','ee','oo','uu','yy','ää','öö']:
                replace_with = random.choice([word + 'pi', word[:-1] + 'vi'])
                verse = remove_extra_material(verse.replace(word, replace_with)).capitalize()

        for word in verse.split():
            word = remove_extra_material(word)
            if len(word) > 1: 
                if len(word) > 2:
                # if word[-1] in vowels and word[-2] not in vowels and len(word) > 2:
                    if contains_back_vowels(word):
                        return remove_extra_material(verse.replace(word, word + 'pa')).capitalize()
                    else:
                        return remove_extra_material(verse.replace(word, word + 'pä')).capitalize()

    if count_syllables(verse) == 6 and not markov_seq:
        verse = add_two_syllables_in_front(verse).capitalize()

    if count_syllables(verse) == 7:
        for word in verse.split():
            word = remove_extra_material(word)
            if word[-2:] in ['aa','ee','oo','uu','yy','ää','öö']:
                replace_with = random.choice([word + 'pi', word[:-1] + 'vi'])
                return remove_extra_material(verse.replace(word, replace_with)).capitalize()

        for word in verse.split():
            word = remove_extra_material(word)
            if len(word) > 1:
                if len(word) > 2:
                # if word[-1] in vowels and word[-2] not in vowels and len(word) > 2:
                    if contains_back_vowels(word):
                        return remove_extra_material(verse.replace(word, word + 'pa')).capitalize()
                    else:
                        return remove_extra_material(verse.replace(word, word + 'pä')).capitalize()
    
    return remove_extra_material(verse).capitalize()


def markov_verse(noun, max_length):
    '''
    Uses a markov model trained with Kalevala and other poems in kalevala meter to create a sequence from input.
    A fallback markov model trained with yle news corpus is used after using 15000 times with the poem markov model.
    When the output candidate is in Kalevala meter (checked with finmeter), function returns it as a string.    
    '''

    i = 0
    while True:

        # Timeout if nothing has been found after 160 000 tries.
        if i > 160000:
            print("\nTimeout. Try again with other words.")
            exit()

        if i < 15000:
            verse_candidate = markov(max_length=max_length, reply_to=noun, reply_mode=ReplyMode.END)
        else:
            verse_candidate = fallback_markov(max_length=max_length, reply_to=noun, reply_mode=ReplyMode.END)
        verse_candidate = remove_extra_material(verse_candidate)

        if has_monosyllabic_word(verse_candidate):
            if i > 22500 and i % 500 == 0:
                verse_candidate = fallback_markov(max_length=max_length, reply_to=random.choice(verse_candidate.split()), reply_mode=ReplyMode.END)

        verse_candidate = fix_syllables(verse_candidate, True)
        
        try:
            meter_result = finmeter.analyze_kalevala(verse_candidate)
        except Exception as e:
            pass

        if meter_result[0]['base_rule']['result'] == True and count_syllables(verse_candidate) == 8:
            markov_sequence = verse_candidate
            break

        if i > 25000  and i % 1000 == 0:
            if len(verse_candidate.split(" ")) > 1:
                noun = random.choice(verse_candidate.split())
                noun = remove_extra_material(fallback_markov(max_length=2, reply_to=noun, reply_mode=ReplyMode.END))
            else:
                noun = remove_extra_material(fallback_markov(max_length=2, reply_to=noun, reply_mode=ReplyMode.END))
                noun = random.choice(noun.split(" "))
        i += 1

    return markov_sequence


def create_verb_probabilities(usr_input):
    ''' Uses the first input noun to find a verbs that are semantically similar. Outputs verb candidates and their probability distribution. '''

    lemmas = tokenize_and_lemmatize(usr_input)
    input_posses = get_pos_template(lemmas)
    # print("Input POSes: " + input_posses + "\n")

    # If both input words are noun. Other alternatives are not implemented.
    if input_posses == 'NN':
        lemma_dict = {'subject':lemmas[0], 'object':lemmas[1]}
        verse = []

        # Loop through both lemmas and inflect them depending on their syntactic role
        for lemma in lemmas:
            # print_some_input_info(lemma) # FOR DEBUGGING

            for analysis in uralicApi.analyze(lemma, "fin"):
                ms_desc = analysis[0].lstrip(analysis[0].split('+')[0])
                # print("Analysis of the lemma: " + lemma + ms_desc + "\n") # FOR DEBUGGING
                if ms_desc[1] == 'N':
                    if lemma == lemma_dict['subject']:
                        generated = uralicApi.generate(lemma + "+N+Sg+Nom", "fin")
                    if lemma == lemma_dict['object']:
                        generated = uralicApi.generate(lemma + "+N+Sg+Gen", "fin")

            if len(generated) > 0:
                verse.append(generated[0][0])
            else:
                print("Try with other words.")

            # If the lemma is subject, choose a verb using its word relations. There's probably a better alternative for this.
            if lemma == lemma_dict['subject']:
                word = semfi.get_word(lemma, "N", "fin")
                while True:
                    try:
                        relations = semfi.get_by_relation(word, "dobj", "fin", sort=True)
                        break
                    except Exception as e:
                        print("At least one of the input words was not recognized, try with other words.\n\n" + e)
                        exit()

                verbs_and_probs = []
                for relation in relations:
                    try:
                        if relation['word2']['pos'] == 'V':
                            inflected_form = uralicApi.generate(relation['word2']['word'] + "+V+Act+Ind+Prs+Sg3", "fin")[0][0]
                            first_syllable = finmeter.hyphenate(inflected_form).split("-")[0]
                            if count_syllables(inflected_form) == 2 and not finmeter.is_short_syllable(first_syllable):
                                verbs_and_probs.append((relation['word2']['word'], relation['word2']['frequency']))
                    except:
                        pass

                # Sort the verb by frequency (descending order) and get rid of the top 5% frequent and the half that is least frequent
                verbs_and_probs = sorted(verbs_and_probs, key=lambda x: x[-1], reverse=True)[round(((len(verbs_and_probs)/100)*5)) : round(((len(verbs_and_probs)/100)*50))]
                if len(verbs_and_probs) == 0:
                    print("Try with other words.")
                    exit()

                else:
                    # Normalize the probabilities and choose the verb randomly 
                    verb_candidates, probability_distribution = map(list, zip(*verbs_and_probs))
                    probability_distribution = np.array(np.array(probability_distribution) / sum(probability_distribution))
        
        return verb_candidates, probability_distribution, lemmas, lemma_dict, verse
                    

def create_first_verse(verb_candidates, probability_distribution, lemmas, verse):
    ''' Picks a verb from the candidates and then creates and outputs the first verse of the stanza '''

    draw = choice(verb_candidates, 1, p=probability_distribution)
    verb = uralicApi.generate(draw[0]+"+V+Act+Ind+Prs+Sg3", "fin")[0][0]
    verse = " ".join(verse)
    # print('verse: ' + verse)
    # verse += " " + verb
    verse = verse.replace(verse.split(" ")[0], verse.split(" ")[0] + " " + verb  + " ")

    return fix_syllables(verse, False)


def print_some_input_info(lemma):
    # This is just for debugging, will delete later.
    print("FOR DEBUGGING:")
    tavutettu = finmeter.hyphenate(lemma)
    syllables = [finmeter.is_short_syllable(tavu) for tavu in tavutettu.split("-")]
    tavuString, tavuja = "", 0
    for isShort in syllables:
        tavuString += "lyhyt-" if isShort else "pitkä-"
        tavuja += 1
    tavuString = tavuString.strip("-")
    print("Syllables: " + tavutettu + " | Syllable lengths: " + tavuString + " | Number of syllables: " + str(tavuja))


def main():
    usr_input = input("Enter two nouns (separated by space): ")
    print("\nGenerating a poem...")
    verb_candidates, probability_distribution, lemmas, lemma_dict, verse = create_verb_probabilities(usr_input)

    first_verse = create_first_verse(verb_candidates, probability_distribution, lemmas, verse)
    print("\n" + first_verse)
    print(markov_verse(first_verse.split()[-2] + " " + first_verse.split()[-1], 3))
    print(markov_verse(lemma_dict['subject'], 3))
    print(markov_verse(lemma_dict['object'], 3))
    print(markov_verse(random.choice(["miten", "kuinka", "vaikka", "jospa", "voiko"]) + " " + lemma_dict['subject'], 3))

    first_verse = create_first_verse(verb_candidates, probability_distribution, lemmas, verse)
    print("\n" + first_verse)
    print(markov_verse(first_verse.split()[-2] + " " + first_verse.split()[-1], 3))
    print(markov_verse(lemma_dict['subject'], 3))
    print(markov_verse(lemma_dict['object'], 3))
    print(markov_verse(random.choice(["miten", "kuinka", "vaikka", "jospa", "voiko"]) + " " + lemma_dict['object'], 3))
    

if __name__ == '__main__':
    main()
