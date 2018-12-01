# TRANSCRIPT SEGMENTATION
import re
import matplotlib.pyplot as plt
import numpy as np
from utils import load_kathy_corpus

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer 

class HearstTextTiler:
    """ Source code from nltk.tokenize. Edited slightly for case usage, some efficiency.
    Implements Hearst TextTiling for text segmentation: http://www.aclweb.org/anthology/J97-1003
    Can also summarize the segmentations.
    
    Usage:
        >> texttiler = HearstTextTiler(w=100, k=40)
        >> plot_data, segmented_text = texttiler(document, plot = True)
        >> summaries = texttiler.summarize(segmented_text, output = True)
    """

    def __init__(self,
                 w=100, # Pseudosentence size
                 k=40, # Stride size
                 stopwords=None, 
                 smoothing_width=2,
                 demo_mode=False):

        if stopwords is None:
            from nltk.corpus import stopwords
            stopwords = stopwords.words('english')
        self.__dict__.update(locals())
        del self.__dict__['self']
        
    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, document, plot = False):
        """ Return a tokenized copy of *text*, where each "token" represents a separate topic. Plots results for visualization if desired.
            Input: list of documents where each element of the list is a sentence/utterance/paragraph.
            Output: tuple containing plot info for visualization, text segmentation blocks where sentence splits marked by " |"
            """
                
        # TOKENIZE
        text = '\n\n\t '.join(document.text)
        cleaned_text = self._clean_text(text)
        nopunct_par_breaks = self._mark_paragraph_breaks(cleaned_text)
        tokseqs = self._divide_to_tokensequences(cleaned_text)

        # Filter stopwords
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)

        # SCORE LEXICAL
        gap_scores = self._block_comparison(tokseqs, token_table)
        smooth_scores = self._smooth_scores(gap_scores)
        
        # IDENTIFY BOUNDARIES
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)
        paragraph_breaks = self._mark_paragraph_breaks(text)
        
        # Normalize boundaries so that we split at the end of original sentences in the output
        normalized_boundaries = self._normalize_boundaries(text, segment_boundaries, paragraph_breaks)
        
        # PREPARE OUTPUT
        segmented_text = []
        prevb = 0

        for b in normalized_boundaries: # take boundaries 
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b

        if prevb < len(cleaned_text): # append any text that may be remaining
            segmented_text.append(text[prevb:])

        if not segmented_text: # if no segmentations (really short or homogeneous text), return text.
            segmented_text = [text]
            
        if plot: # plot if desired
            self._plot(gap_scores, smooth_scores, depth_scores, segment_boundaries)
        
        segmented_text = [re.sub("\n\n\t ", " | ", segment) for segment in segmented_text] # clean output
        return (gap_scores, smooth_scores, depth_scores, segment_boundaries), segmented_text

    def summarize(self, segmented_text, output = True):
        tokenizer, summarizer = Tokenizer("english"), LsaSummarizer()
        summaries = []
        for segment in segmented_text:
            parser = PlaintextParser.from_string(segment, tokenizer)
            summary = summarizer(parser.document, 1)
            if output:
                print(' '.join(summary[0].words), '\n')
            summaries.append(' '.join(summary[0].words))
            
        return summaries
    
    def _clean_text(self, text):
        """ Remove everything but whitespace, the alphabet. Separate apostrophes for stopwords """
        
        clean_text = re.sub(r"[^a-z\s']", '', text.lower())
        clean_text = re.sub(r"[']+", ' ', clean_text)
        return text

    def _mark_paragraph_breaks(self, text):
        """Identifies indented text or line breaks as the beginning of paragraphs"""
        
        MIN_PARAGRAPH = 100
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start()-last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks
    
    def _divide_to_tokensequences(self, text):
        """ Divides the text into 'pseudosentences' of fixed size """
        
        w = self.w
        wrdindex_list = []
        matches = re.finditer("\w+", text)
        for match in matches:
            wrdindex_list.append((match.group(), match.start()))
        return [TokenSequence(i/w, wrdindex_list[i:i+w])
                for i in range(0, len(wrdindex_list), w)]

    def _create_token_table(self, token_sequences, par_breaks):
        """ Creates a table of TokenTableFields """
        
        token_table = {}
        current_par = 0
        current_tok_seq = 0
        pb_iter = par_breaks.__iter__()
        current_par_break = next(pb_iter)
        if current_par_break == 0:
            try:
                current_par_break = next(pb_iter) #skip break at 0
            except StopIteration:
                raise ValueError(
                    "No paragraph breaks were found(text too short perhaps?)"
                    )
        for ts in token_sequences:
            for word, index in ts.wrdindex_list:
                try:
                    while index > current_par_break:
                        current_par_break = next(pb_iter)
                        current_par += 1
                except StopIteration:
                    #hit bottom
                    pass

                if word in token_table:
                    token_table[word].total_count += 1

                    if token_table[word].last_par != current_par:
                        token_table[word].last_par = current_par
                        token_table[word].par_count += 1

                    if token_table[word].last_tok_seq != current_tok_seq:
                        token_table[word].last_tok_seq = current_tok_seq
                        token_table[word]                                .ts_occurences.append([current_tok_seq,1])
                    else:
                        token_table[word].ts_occurences[-1][1] += 1
                else: #new word
                    token_table[word] = TokenTableField(first_pos=index,
                                                        ts_occurences= \
                                                          [[current_tok_seq,1]],
                                                        total_count=1,
                                                        par_count=1,
                                                        last_par=current_par,
                                                        last_tok_seq= \
                                                          current_tok_seq)

            current_tok_seq += 1

        return token_table
    
    def _block_comparison(self, tokseqs, token_table):
        "Implements the block comparison method"

        gap_scores = []
        numgaps = len(tokseqs)-1

        for curr_gap in range(numgaps):
            score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
            score = 0.0
            #adjust window size for boundary conditions
            if curr_gap < self.k-1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps-self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k

            b1 = [ts.index
                  for ts in tokseqs[curr_gap-window_size+1 : curr_gap+1]]
            b2 = [ts.index
                  for ts in tokseqs[curr_gap+1 : curr_gap+window_size+1]]

            for t in token_table:
                score_dividend += self._blk_frq(t, b1, token_table)*self._blk_frq(t, b2, token_table)
                score_divisor_b1 += self._blk_frq(t, b1, token_table)**2
                score_divisor_b2 += self._blk_frq(t, b2, token_table)**2
            try:
                score = score_dividend/((score_divisor_b1*
                                                 score_divisor_b2)**0.5)
            except ZeroDivisionError:
                pass # score += 0.0

            gap_scores.append(score)

        return gap_scores
    
    def _blk_frq(self, tok, block, token_table):
        """ Count occurrences of a token in a block """
        
        ts_occs = filter(lambda o: o[0] in block,
                         token_table[tok].ts_occurences)
        freq = sum([tsocc[1] for tsocc in ts_occs])
        return freq
    
    def _smooth_scores(self, gap_scores):
        """ Wraps the SciPy smooth function """
        
        return list(self._smooth(np.array(gap_scores[:]), window_len = self.smoothing_width+1))
    
    def _smooth(self, x, window_len=11, window='flat'):
        """ Source code fom SciPy: window smoothing function """
        
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        elif x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        elif window_len < 3:
            return x
        elif window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]

        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w/w.sum(), s, mode='same')

        return y[window_len-1:-window_len+1]
    
    def _identify_boundaries(self, depth_scores):
        """Identifies boundaries at the peaks of similarity score differences"""

        boundaries = np.zeros(len(depth_scores))#[0 for x in depth_scores]

        avg = sum(depth_scores)/len(depth_scores)
        stdev = np.std(depth_scores)

        cutoff = avg-stdev/2.0

        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x:x[0]>cutoff, depth_tuples))

        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp: #undo if there is a boundary close already
                if dt[1] != dt2[1] and abs(dt2[1]-dt[1]) < 4                        
                and boundaries[dt2[1]] == 1:
                    boundaries[dt[1]] = 0
        return boundaries

    def _depth_scores(self, scores):
        """Calculates the depth of each gap, i.e. the average difference
        between the left and right peaks and the gap's score"""

        depth_scores = [0 for _ in scores]
        #clip boundaries: this holds on the rule of thumb(my thumb)
        #that a section shouldn't be smaller than at least 2
        #pseudosentences for small texts and around 5 for larger ones.

        clip = int(min(max(len(scores)/10, 2), 5))
        index = clip

        for gapscore in scores[clip:-clip]:
            lpeak = gapscore
            for score in scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = lpeak + rpeak - 2 * gapscore
            index += 1

        return depth_scores
    
    def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
        """Normalize the boundaries identified to the original text's
        paragraph breaks"""

        norm_boundaries = []
        char_count, word_count, gaps_seen = 0, 0, 0
        seen_word = False

        for char in text:
            char_count += 1
            if char in " \t\n" and seen_word:
                seen_word = False
                word_count += 1
            if char not in " \t\n" and not seen_word:
                seen_word=True
            if gaps_seen < len(boundaries) and word_count > (max(gaps_seen*self.w, self.w)):
                if boundaries[gaps_seen] == 1:
                    #find closest paragraph break
                    best_fit = len(text)
                    for br in paragraph_breaks:
                        if best_fit > abs(br-char_count):
                            best_fit = abs(br-char_count)
                            bestbr = br
                        else:
                            break
                    if bestbr not in norm_boundaries: #avoid duplicates
                        norm_boundaries.append(bestbr)
                gaps_seen += 1

        return norm_boundaries
    
    def _plot(self, gap_scores, smooth_scores, depth_scores, segment_boundaries):
        """ Plots segmentation results for visualization purposes. """
        
        plt.style.use('ggplot')
        plt.figure(figsize=(40, 20))
        plt.xlabel("Sentence Gap index", fontsize = 36)
        plt.xticks(fontsize = 25)
        plt.ylabel("Gap Scores", fontsize = 36)
        plt.yticks(fontsize = 25)
        plt.plot(range(len(smooth_scores)), smooth_scores, label="Smoothed Gap scores", color = 'green')
        plt.plot(range(len(depth_scores)), depth_scores, label="Depth scores", color = 'orange')
        plt.stem(range(len(segment_boundaries)), segment_boundaries)
        plt.legend(loc = 'upper right', fontsize = 25)
        plt.show()
        print('Num. segments: {0}'.format(sum(segment_boundaries)+1))
    
class TokenSequence:
    """ A token list with its original length and its index
    Source code from nltk.tokenize """
    def __init__(self,
                 index,
                 wrdindex_list,
                 original_length=None):
        original_length=original_length or len(wrdindex_list)
        self.__dict__.update(locals())
        del self.__dict__['self']

class TokenTableField(object):
    """ A field in the token table holding parameters for each token, used later in the process
    Source code from nltk.tokenize """
    def __init__(self,
                 first_pos,
                 ts_occurences,
                 total_count=1,
                 par_count=1,
                 last_par=0,
                 last_tok_seq=None):
        self.__dict__.update(locals())
        del self.__dict__['self']
