import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_batch = []
        blank = 0
        path_probs = []
        sym_len, seq_len, batch_size = y_probs.shape
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        #return decoded_path, path_prob
        for b in range(batch_size):
            decoded_path = []
            path_prob = 1
            for i in range(seq_len):
                idx = np.argmax(y_probs[:, i, b])
                path_prob *= y_probs[idx, i, b]

                if idx != 0:
                    symbol = self.symbol_set[idx - 1]
                    if len(decoded_path) == 0 or symbol != decoded_path[-1]:
                        decoded_path.append(symbol)
            path_probs.append(path_prob)
            decoded_batch.append(''.join(decoded_path))
        return decoded_batch[0], path_probs[0]
        

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        sym_len, seq_len, batch_size = y_probs.shape
        
        # TODO:
        # Implement the beam search decoding algorithm. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols.
        # 3. Merging paths that end in the same symbol.
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths.
        # 5. After iterating through all time steps, selecting the best path
        #    and its score.
        
        #ctc decoder for batchsize = 1
        blankPathScore = {"": y_probs[0, 0, 0]}
        pathScore = {}
        for i, symbol in enumerate(self.symbol_set):
            pathScore[symbol] = y_probs[i + 1, 0, 0]

        for t in range(1, T):
            allPathScores = [(path, score) for path, score in blankPathScore.items()]
            allPathScores += [(path, score) for path, score in pathScore.items()]
            allPathScores.sort(key=lambda x: x[1], reverse=True)
            cutoff = allPathScores[self.beam_width][1] if self.beam_width < len(allPathScores) else allPathScores[-1][1]

            pruned_paths_blank = {p: s for p, s in blankPathScore.items() if s > cutoff}
            pruned_paths_symbol = {p: s for p, s in pathScore.items() if s > cutoff}

            newBlankScore = {}
            for path, score in pruned_paths_blank.items():
                newBlankScore[path] = score * y_probs[0, t, 0]
            for path, score in pruned_paths_symbol.items():
                if path in newBlankScore:
                    newBlankScore[path] += score * y_probs[0, t, 0]
                else:
                    newBlankScore[path] = score * y_probs[0, t, 0]

            newPathScore = {}
            for path, score in pruned_paths_blank.items():
                for i, symbol in enumerate(self.symbol_set):
                    new_path = path + symbol
                    if new_path in newPathScore:
                        newPathScore[new_path] += score * y_probs[i + 1, t, 0]
                    else:
                        newPathScore[new_path] = score * y_probs[i + 1, t, 0]

            for path, score in pruned_paths_symbol.items():
                for i, symbol in enumerate(self.symbol_set):
                    if symbol == path[-1]:
                        new_path = path
                    else:
                        new_path = path + symbol
                    if new_path in newPathScore:
                        newPathScore[new_path] += score * y_probs[i + 1, t, 0]
                    else:
                        newPathScore[new_path] = score * y_probs[i + 1, t, 0]
            blankPathScore = newBlankScore
            pathScore = newPathScore    
        mergedPathScores = {}
        for path, score in blankPathScore.items():
            if path in mergedPathScores:
                mergedPathScores[path] += score
            else:
                mergedPathScores[path] = score  
        for path, score in pathScore.items():
            if path in mergedPathScores:
                mergedPathScores[path] += score
            else:
                mergedPathScores[path] = score
        # Select best path      
        bestPath = max(mergedPathScores, key=mergedPathScores.get)
        return bestPath, mergedPathScores
            
        
