import finmeter, random, lwvlib, json
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

def count_syllables(verse):
    # Count syllables in a verse
    tokenizer = RegexpTokenizer(r'\w+')
    n_syllables = 0
    for word in tokenizer.tokenize(verse):
        n_syllables += len(finmeter.hyphenate(word).split("-"))

    return n_syllables


def most_frequent(List):
    # Return the most frequent element in a list
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 


def tokenize_and_lemmatize(words):
    # Tokenize and lemmatize input. Returns a list of lemmas.
    tokenizer = RegexpTokenizer(r'\w+')
    lemmas = []

    for word in tokenizer.tokenize(words):
        for i, analysis in enumerate(uralicApi.analyze(word.lower(), "fin")):
            if analysis[0].split('+', 1)[1] == 'N+Sg+Nom':
                lemmas.append(analysis[0].split('+')[0])

    if len(lemmas) < 2:
        print("Both words were not recognized. The program will close.")
        exit()

    return lemmas


def markov_verse(name):
    '''
    Use a markov model trained with Kalevala to create a sequence from input.
    When the output candidate is in Kalevala meter (checked with finmeter), function returns it as a string.    
    '''

    while True:
        verse_candidate = markov(max_length=4, reply_to=name, reply_mode=ReplyMode.END)
        try:
            meter_result = finmeter.analyze_kalevala(verse_candidate)
            if meter_result[0]['base_rule']['result'] == True: #and count_syllables(verse_candidate) == 8:
                markov_sequence = verse_candidate
                break
        except:
            pass

    return markov_sequence


def get_pos_template(lemmas):
    # Return the POSses of each input word. E.g. "NN"
    # Currently only searches for NN.
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

def add_two_syllables_in_front(verse):
    return verse[:2] + random.choice(['k','t','p','s']) + uralicApi.analyze(verse.split(" ")[0], "fin")[0][0].split('+')[0][-1] + " " + verse


def fix_syllables(verse):
    # This function should do something when the amount of syllables is not correct. It is not yet implemented.
    if count_syllables(verse) == 5:
        verse = verse.replace(verse.split()[1], verse.split()[1] + "pi", 1)
        verse = add_two_syllables_in_front(verse)
        if verse.split()[0][-1] not in vowels:
            verse = verse.replace(verse.split()[0][-1], "a", 1)

    if count_syllables(verse) == 6:
        verse = add_two_syllables_in_front(verse)
        if verse.split()[0][-1] not in vowels:
            verse = verse.replace(verse.split()[0][-1], "a", 1)

    if count_syllables(verse) == 7:
        verse = verse.replace(verse.split()[1], verse.split()[1] + "pi", 1)
    
    return verse


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
    usr_input = input("Enter two nouns: ")
    lemmas = tokenize_and_lemmatize(usr_input)
    input_posses = get_pos_template(lemmas)
    print("Input POSes: " + input_posses + "\n")

    # If both input words are noun. Other alternatives are not implemented.
    if input_posses == 'NN':
        lemma_dict = {'subject':lemmas[0], 'object':lemmas[1]}
        verse = []

        # Loop through both lemmas and inflect them depending on their syntactic role
        for lemma in lemmas:
            print_some_input_info(lemma) # FOR DEBUGGING

            for analysis in uralicApi.analyze(lemma, "fin"):
                ms_desc = analysis[0].lstrip(analysis[0].split('+')[0])
                print("Analysis of the lemma: " + lemma + ms_desc + "\n")
                if ms_desc[1] == 'N':
                    if lemma == lemma_dict['subject']:
                        generated = uralicApi.generate(lemma + "+N+Sg+Nom", "fin")
                    if lemma == lemma_dict['object']:
                        # generated = uralicApi.generate(lemma + "+N+Sg+" + random.choice(["Gen","Par"]), "fin") # Partitiivi ei oikein toiminut.
                        generated = uralicApi.generate(lemma + "+N+Sg+Gen", "fin")

            if len(generated) > 0:
                verse.append(generated[0][0])
            else:
                print("Nothing found. Fix this.")

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
                verb_candidates, probability_distribution = map(list, zip(*verbs_and_probs))

                # Normalize the probabilities and choose the verb randomly 
                probability_distribution = np.array(np.array(probability_distribution) / sum(probability_distribution))
                draw = choice(verb_candidates, 1, p=probability_distribution)
                lemma_dict['verb'] = draw[0]
                verse.append(uralicApi.generate(draw[0]+"+V+Act+Ind+Prs+Sg3", "fin")[0][0])

        # Tämä ajetaan kun input on käyty läpi (sanat taivutettu ja verbi etsitty)
        verse = " ".join(verse)
        verse = fix_syllables(verse)
        print("Syllables in first verse: " + str(count_syllables(verse)))
        first_verse_analysis = finmeter.analyze_kalevala(verse)

        # If first verse not in kalevala meter, print the error.
        if first_verse_analysis[0]['base_rule']['result'] == False:
            print("Errors in first verse: " + first_verse_analysis[0]['base_rule']['message'])

    else:
        print("Input words were not recognized.")
        exit()

    # Laitoin tästä eteenpäin testiksi vain erilaisia komboja syötteeksi markov-funktiolle. Se sallii nyt myös 10-tavuiset säkeet, 8-tavuisia sillä tuntuu
    # olevan vaikeuksia löytää ja se jää junnaamaan, vaikka se olisikin helppo laittaa kriteeriksi. Markov-malli ei löydä kovin monelle sanalle jatkoa
    # eli sille pitää kehitellä joku fallback-systeemi tai käyttää Kalevalan lisäksi myös jotain muuta korpusta opetuksessa.

    print("\n" + verse)
    print(markov_verse(verse.split()[-2] + " " + verse.split()[-1]))
    print(markov_verse(lemma_dict['subject']))
    print(markov_verse(lemma_dict['object']))
    print(markov_verse(random.choice(["miten", "kuinka", "vaikka", "jospa"]) + " " + lemma_dict['subject']))
    

if __name__ == '__main__':
    main()
