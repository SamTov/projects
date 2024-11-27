# Universal Dependencies for Croatian

### Training set.

Contains 7,689 sentences (169,283 tokens) from three sources:

1. Sentences 0001-3557: Newspaper text from the [Southeast European Times](http://en.wikipedia.org/wiki/Southeast_European_Times) news website, obtained from the [SETimes parallel corpus](http://nlp.ffzg.hr/resources/corpora/setimes/). This part of the treebank is built on top of the [SETimes.HR dependency treebank of Croatian](https://github.com/ffnlp/sethr);
2. Sentences 3558-5792: Text from various [Croatian web sources](http://nl.ijs.si/isjt14/proceedings/isjt2014_10.pdf).
3. Sentences 5793-7689: Croatian news web sources.

### Development set.

Contains 600 sentences (14,533 tokens) from two sources:

1. 001-200: newspaper text from the Croatian SETimes,
2. 201-600: Croatian news web sources.

### Test set.

Contains 600 sentences (13,228 tokens) from three sources:

1. sentences 001-100: newspaper text,
2. sentences 101-200: Wikipedia,
3. sentences 201-297: web sources, and
4. sentences 298-600: Croatian news web sources.

### Details

Sentence and word segmentation was manually checked. The treebank does not include multiword tokens. No language-specific features and relations were used. The POS tags and features were converted from [Multext East v4](http://nlp.ffzg.hr/data/tagging/msd-hr.html) and manually checked. The syntactic annotation was done manually.

When using the Croatian UD treebank, please cite the following paper:

* Željko Agić and Nikola Ljubešić. 2015. [Universal Dependencies for Croatian (that work for Serbian, too).](http://aclweb.org/anthology/W/W15/W15-5301.pdf). In Proc. BSNLP, pp. 1--8 ([bib](http://aclweb.org/anthology/W/W15/W15-5301.bib)).

See file LICENSE.txt for further licensing information.

### Changelog

2017-02-15

* converted to UD v2 standard
  * nmod vs. obl under non-verbal predicates should be checked manually (see the ToDo attribute in the MISC column)
  * by UD guidelines, reflexive pronouns with inherently reflexive verbs are now attached as expl:pv, not compound
  * adverbial participles (converbs) are marked by VerbForm=Conv
* a number of enhancements and bug fixes
  * all pronouns, determiners and pronominal adverbs have PronType
  * all verbs have VerbForm; all finite verbs have Mood
  * ordinal numerals are ADJ like elsewhere in UD, not NUM (but they keep NumType=Ord)
  * relative pronouns, determiners and adverbs are not attached as mark (subordinating conjunctions keep the mark relation)
  * possessive adjectives and determiners are amod and det respectively; not nmod
  * coordinating conjunctions at the beginning of sentence are attached as cc, not discourse

2017-02-09

* added new ud v1 sentences from news-hr to dev, test, and train set: 2600 sentences, out of which the last 703 went to dev (400) and test (303), and the remainder to the train set

2016-10-31

* added 2235 new sentences to the training set, and 97 new sentences to the test set, from various Croatian web sources



=== Machine-readable metadata =================================================
Documentation status: stub
Data source: semi-automatic
Data available since: UD v1.1
License: CC BY-SA 4.0
Genre: news web wiki
Contributors: Agić, Željko; Ljubešić, Nikola; Zeman, Daniel
Contact: zeljko.agic@gmail.com
===============================================================================

