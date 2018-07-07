import math
from collections import Counter, defaultdict

import random


class KneserNeyLM:

    def __init__(self, highest_order, ngrams, start_pad_symbol='<s>',
                 end_pad_symbol='</s>'):
        """
        Constructor for KneserNeyLM.

        Params:
            highest_order [int] The order of the language model.
            ngrams [list->tuple->string] Ngrams of the highest_order specified.
                Ngrams at beginning / end of sentences should be padded.
            start_pad_symbol [string] The symbol used to pad the beginning of
                sentences.
            end_pad_symbol [string] The symbol used to pad the beginning of
                sentences.
        """
        self.highest_order = highest_order
        self.start_pad_symbol = start_pad_symbol
        self.end_pad_symbol = end_pad_symbol
        self.lm = self.train(ngrams)

    def train(self, ngrams):
        """
        Train the language model on the given ngrams.

        Params:
            ngrams [list->tuple->string] Ngrams of the highest_order specified.
        """
        kgram_counts = self._calc_adj_counts(Counter(ngrams))
        probs = self._calc_probs(kgram_counts)
        return probs

    def highest_order_probs(self):
        return self.lm[0]

    def _calc_adj_counts(self, highest_order_counts):
        """
        Calculates the adjusted counts for all ngrams up to the highest order.

        Params:
            highest_order_counts [dict{tuple->string, int}] Counts of the highest
                order ngrams.

        Returns:
            kgrams_counts [list->dict] List of dict from kgram to counts
                where k is in descending order from highest_order to 0.
        """
        kgrams_counts = [highest_order_counts]
        for i in range(1, self.highest_order):
            last_order = kgrams_counts[-1]
            new_order = defaultdict(int)
            for ngram in last_order.keys():
                suffix = ngram[1:]
                new_order[suffix] += 1
            kgrams_counts.append(new_order)
        return kgrams_counts

    def _calc_probs(self, orders):
        """
        Calculates interpolated probabilities of kgrams for all orders.
        """
        backoffs = []
        for order in orders[:-1]:
            backoff = self._calc_order_backoff_probs(order)
            backoffs.append(backoff)
        orders[-1] = self._calc_unigram_probs(orders[-1])
        backoffs.append(defaultdict(int))
        self._interpolate(orders, backoffs)
        return orders

    def _calc_unigram_probs(self, unigrams):
        sum_vals = sum(v for v in unigrams.values())
        unigrams = dict((k, math.log(v / sum_vals)) for k, v in unigrams.items())
        return unigrams

    def _calc_order_backoff_probs(self, order):
        num_kgrams_with_count = Counter(
            value for value in order.values() if value <= 4)
        discounts = self._calc_discounts(num_kgrams_with_count)
        prefix_sums = defaultdict(int)
        backoffs = defaultdict(int)
        for key in order.keys():
            prefix = key[:-1]
            count = order[key]
            prefix_sums[prefix] += count
            discount = self._get_discount(discounts, count)
            order[key] -= discount
            backoffs[prefix] += discount
        for key in order.keys():
            prefix = key[:-1]
            order[key] = math.log(order[key] / prefix_sums[prefix])
        for prefix in backoffs.keys():
            backoffs[prefix] = math.log(backoffs[prefix] / prefix_sums[prefix])
        return backoffs

    def _get_discount(self, discounts, count):
        if count > 3:
            return discounts[3]
        return discounts[count]

    def _calc_discounts(self, num_with_count):
        """
        Calculate the optimal discount values for kgrams with counts 1, 2, & 3+.
        """
        common = num_with_count[1] / (num_with_count[1] + 2 * num_with_count[2])
        # Init discounts[0] to 0 so that discounts[i] is for counts of i
        discounts = [0]
        for i in range(1, 4):
            if num_with_count[i] == 0:
                discount = 0
            else:
                discount = (i - (i + 1) * common
                            * num_with_count[i + 1] / num_with_count[i])
            discounts.append(discount)
        if any(d for d in discounts[1:] if d <= 0):
            raise Exception(
                '***Warning*** Non-positive discounts detected. '
                'Your dataset is probably too small.')
        return discounts

    def _interpolate(self, orders, backoffs):
        """
        """
        for last_order, order, backoff in zip(
                reversed(orders), reversed(orders[:-1]), reversed(backoffs[:-1])):
            for kgram in order.keys():
                prefix, suffix = kgram[:-1], kgram[1:]
                order[kgram] += last_order[suffix] + backoff[prefix]

    def logprob(self, ngram):
        for i, order in enumerate(self.lm):
            if ngram[i:] in order:
                return order[ngram[i:]]
        return 0

    def score_sent(self, sent):
        """
        Return log prob of the sentence.

        Params:
            sent [tuple->string] The words in the unpadded sentence.
        """
        padded = (
                (self.start_pad_symbol,) * (self.highest_order - 1) + sent +
                (self.end_pad_symbol,))
        sent_logprob = 0
        for i in range(len(sent) - self.highest_order + 1):
            ngram = sent[i:i + self.highest_order]
            sent_logprob += self.logprob(ngram)
        return sent_logprob

    def generate_sentence(self, min_length=4):
        """
        Generate a sentence using the probabilities in the language model.

        Params:
            min_length [int] The mimimum number of words in the sentence.
        """
        sent = []
        probs = self.highest_order_probs()
        while len(sent) < min_length + self.highest_order:
            sent = [self.start_pad_symbol] * (self.highest_order - 1)
            # Append first to avoid case where start & end symbal are same
            sent.append(self._generate_next_word(sent, probs))
            while sent[-1] != self.end_pad_symbol:
                sent.append(self._generate_next_word(sent, probs))
        sent = ' '.join(sent[(self.highest_order - 1):-1])
        return sent

    def _get_context(self, sentence):
        """
        Extract context to predict next word from sentence.

        Params:
            sentence [tuple->string] The words currently in sentence.
        """
        return sentence[(len(sentence) - self.highest_order + 1):]

    def _generate_next_word(self, sent, probs):
        context = tuple(self._get_context(sent))
        pos_ngrams = list(
            (ngram, logprob) for ngram, logprob in probs.items()
            if ngram[:-1] == context)
        # Normalize to get conditional probability.
        # Subtract max logprob from all logprobs to avoid underflow.
        _, max_logprob = max(pos_ngrams, key=lambda x: x[1])
        pos_ngrams = list(
            (ngram, math.exp(prob - max_logprob)) for ngram, prob in pos_ngrams)
        total_prob = sum(prob for ngram, prob in pos_ngrams)
        pos_ngrams = list(
            (ngram, prob / total_prob) for ngram, prob in pos_ngrams)
        rand = random.random()
        for ngram, prob in pos_ngrams:
            rand -= prob
            if rand < 0:
                return ngram[-1]
        return ngram[-1]
