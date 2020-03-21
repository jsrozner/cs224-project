"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class Paraphraser(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors. (will be updated during train time)
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(Paraphraser, self).__init__()
        # We load embeddings from a glove vector file.
        # embedding, drop, projection (linear), highway layer - todo: do we want all of these or just embedding?
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        # todo: we also need to add a do not replace vector that needs to be trained
        # initialize a gumbel softmax (or just use the nn.functional)

    def forward(self, cw_idxs, qw_idxs, qw_to_phrases, rw_idxs):
        """
        :param cw_idxs: The words in the context
        :param qw_idxs: The words in the question
        :param qw_to_phrases: breakdown of qw by spans into phrases (specifies how to replace a given word)
        :param rw_idxs:    A matrix of phrasal replacements per phrase type, for each context.

        :return: updated qw_idxs (i.e. a paraphrased question) -- some of the qw_idx will be replaced with
        idxs from cw based on the replacement_idxs
        """
        #c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)


        # add padding to rw_idxs:
        # need to add padding for the replacement_phrases (so total possible replacements is equal)

        # add padding to the qw_to_phrases matrix
        # questions will have variable number of phrases to replace. this pads so that all
        # questions have same number of possible phrasal replacements

        # before padding:
        # pre_pad_rw = (batch_size, num_phrase_types, num_candidate_phrases, length_replacement_phrase)
        # last dimension is a "list" of words (word indices) for a given phrase

        # after padding:
        # (batch_size, num_phrase_types, num_candidate_phrases, max_length_replacement_phrase)
        #       # for each phrase type, this is a matrix of (phrase type, set of phrasal replacements for this type)

        # todo: after padding for the qw_to_phrase mapping
        # (batch_size, max_number_of_phrases_parsed_from_a_question, ....)

        # do embedding lookup for each of the words:
        # (batch_size, num_phrase_types, num_candidate_phrases, max_words_in_replacement_phrase, embedding_hidden_size)

        # take a vector average (over all words in the phrase - over the max_words_in_replacement_dim)
        # don't count pads
        # (batch_size, num_phrase_types, num_candidate_phrases, embedding_hidden_size)

        # append the do not replace (DNR) token for each phrase type
        # (batch_size, num_phrase_types, num_candidate_phrases + 1, embedding_hidden_size)

        # now "unsqueeze" - expand based on the type of the phrase for each question
        # this uses the qw_to_phrases to expand by phrasal type
        # (batch_size, max_num_phrases_in_q, num_candidate_phrases +1, embedding_hidden_size)

        # calculate cosine similarity (todo: maybe take log?) between each replacement phrase and the original phrase
        # this is cosine similarity along the embedding hidden size dimension
        # output: (batch_size, max_num_phrases_in_q, num_candidate_phrases +1)
        # for each phrase in the question, dot product with every candidate replacement phrase at the same index
        # for each question, we have (num_phrases in question) * (num_candidate_phrases +1) dot products

        # sample using gumbel softmax
        # (batch_size, max_num_phrases_in_q, 1) where the last spot has the index of selected candidate_phrase
        # todo: some sort of masking / ignoring of the rows that are padded?

        # conduct the replacement:
        # generate a full lookup table (need to include dnr)
        # pre_pad_rw_with_dnr = append the original word indices from the original phrase (so a DNR index returns the
        # list of original word indices)

        # do the lookup - for each item in (batch_size, max_num_phrases_in_q, 1), where the 1 is the index
        # in the lookup, substitute in the list of actual word tokens
        # gives (batch_size, max_num_phrases_in_q, [list of word tokens])

        # finally concatenate along the phrases dimension (also reduce to max length, or drop)
        # (batch_size) where each entry is [list of all word tokens in the whole paraphrased question])

        # return this final thing as the paraphrased question indices that will be passed into the BiDAF!

        # c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        # q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        # r_emb = self.emb(rw_idxs)


    # todo: implement an annealing function that changes the tau for the model, as we progress

###
# original comment note for the bidaf model
# """Baseline BiDAF model for SQuAD.
# Based on the paper:
# "Bidirectional Attention Flow for Machine Comprehension"
# by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
# (https://arxiv.org/abs/1611.01603).
#
# Follows a high-level structure commonly found in SQuAD models:
# - Embedding layer: Embed word indices to get word vectors.
# - Encoder layer: Encode the embedded sequence.
# - Attention layer: Apply an attention mechanism to the encoded sequence.
# - Model encoder layer: Encode the sequence again.
# - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
# """

# c_mask = torch.zeros_like(cw_idxs) != cw_idxs
# q_mask = torch.zeros_like(qw_idxs) != qw_idxs
# c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
#
# c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
# q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
#
# c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
# q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
#
# att = self.att(c_enc, q_enc,
#                c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
#
# mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
#
# out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
