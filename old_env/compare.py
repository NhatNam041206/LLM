# import time

truth_tex='data/A geometric notion of singularity.vtt'
pred_tex='pred.txt'

with open(truth_tex, 'r', encoding='utf-8') as f:
    truth_lines = f.readlines()

with open(pred_tex, 'r', encoding='utf-8') as f:
    pred_lines = f.readlines()

def extract_word(texts):
    words = []
    punctuations = '''!()-[]{};:"\,<>./?@#$%^&*_~1234567890'''
    for line in texts:
        line = line.replace('\n', ' ')
        if not (line and not line.startswith('NOTE') and not line.startswith('WEBVTT') and not line[0].isdigit()):
            continue
    
        word=''
        for char in line:
            if char not in punctuations and char!=' ':
                word+=char
            if char==' ' and word!='':
                words.append(word.lower())
                word=''

        if word!='' and word.lower() not in words:
            words.append(word.lower())

    return words

truth=extract_word(truth_lines)
pred=extract_word(pred_lines)

print('----------Truth----------')
print("Truth word count:", len(truth))
print('Unique word count:', len(set(truth)))

print('\n----------Pred----------')
print("Predicted word count:", len(pred))
print('Unique word count:', len(set(pred)))

common_words = set(truth) & set(pred)
mismatched_words = set(truth) - set(pred) | set(pred) - set(truth)
print('\n----------Stats----------')
print("Common word count:", len(common_words))
print("Mismatched word count:", len(mismatched_words))
print('\nMismatched words:')
print(mismatched_words)
