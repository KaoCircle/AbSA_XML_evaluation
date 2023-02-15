import xml.etree.ElementTree as ET
import pandas as pd

def xmlReader(filename):
    # XML file will be transferred to dataframe
    # all the opinion aspects in a sentence will be put in a list of dict
    raw = ET.parse(filename)
    root = raw.getroot()
    d_cols = ['text', 'relevance', 'sentiment', "opinions"]
    polar_num = {'negative': 0, 'neutral': 1, 'positive': 2}
    docs = []

    for node in root:
        text = node.findtext('text')
        
        r = node.findtext('relevance').split()[0].casefold()
        relevance = True if r=='true' else False

        s = node.findtext('sentiment').split()[0].casefold()
        sentiment = polar_num[s]
        #id = node.attrib['id']

        ops_xml = node.find('Opinions')
        opinions = []
        if ops_xml is not None:
            for op in ops_xml:
                category = op.attrib['category'].split('#') # split with # character, 0 for main, 1 for sub
                offset_from = int(op.attrib['from'])
                offset_to = int(op.attrib['to'])
                term = op.attrib['target']
                p = op.attrib['polarity'].split()[0]
                #polarity = polar_num[p]
                if term != 'NULL':
                    opinions.append({'from': offset_from, 'to': offset_to, 'term': term,
                        'polarity': p, 'category': category[0], 'subcategory': category[1]})

        docs.append({'text': text, 'relevance': relevance, 'sentiment': sentiment, 'opinions': opinions})

    return pd.DataFrame(docs, columns = d_cols)


def string_offsets(s):
    # split the string by ' '(space)
    # return a list of word begin and word end offset tuples
    words = s.split()
    flag = 0
    offset_list = []
    for w in words:
        w_offset = s.index(w, flag)
        flag = w_offset + len(w)
        offset_list.append((w_offset, flag-1))

    return offset_list


def find_iob(opinions, offsets):
    # return IOB label for opinion target according to offset list
    iob = ['O']*len(offsets)
    flag_b, flag_e = 0, 0
    for op in opinions:
        detected = False
        for id, offset in enumerate(offsets):
            if not detected and op['from']>=offset[0] and op['from']<=offset[1]:
                flag_b = id
                detected = True
            if detected and op['to']<=offset[1]+1:
                flag_e = id
                break
        for j in range(flag_b, flag_e+1):
            iob[j] = 'I-'+op['polarity']
        iob[flag_b] = 'B-'+op['polarity']

    return iob
