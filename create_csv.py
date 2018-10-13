#!usr/bin/python
# coding: utf8

import codecs


def create_csv(ids, f_scores, locations, o_scores):
    """Create the csv file based on the args.

    Each arg should be a list of size n, n being the number
    of images. Then:
    - f_scores[i]: a probability indicating broken or not
    - location[i]: 0 for antérieure, 1 for postérieure
    - o_scores[i]: an array of size 2: [<h prob>, <v prob>]
    """
    res = [u'id,Corne anterieure,Corne posterieure,Fissure,Orientation horizontale,Orientation verticale']
    for m_id, f_score, location, orientations in zip(ids, f_scores, locations, o_scores):
        ant, post = (1, 0) if location == 0 else (0, 1)
        res.append(u'%s,%s,%s,%s,%s,%s' % (m_id, ant, post, f_score, orientations[0], orientations[1]))

    csv = u'\n'.join(res)
    with codecs.open('radioAdvisor.csv', 'w', 'utf-8') as file:
        file.write('\ufeff')  # UTF-8 BOM header
        file.write(csv)


if __name__ == '__main__':
    create_csv(['1'], [0.231], [0], [[0.12, 0.88]])
