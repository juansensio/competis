VOCAB = [
  'PAD',
  'SOS',
  'EOS',
  '(',
  ')',
  '+',
  ',',
  '-',
  '/',
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  'B',
  'C',
  'D',
  'F',
  'H',
  'I',
  'N',
  'O',
  'P',
  'S',
  'T',
  'b',
  'c',
  'h',
  'i',
  'l',
  'm',
  'r',
  's',
  't'
]

def compute_vocab(InChIs):
    special = ['PAD', 'SOS', 'EOS']
    vocab = special + sorted(list({s for InChI in InChIs for s in InChI}))
    return vocab

def t2ix(t, vocab=VOCAB):
    return vocab.index(t)

def encode(InChI):
    tokens = [w for w in InChI]
    ixs = [t2ix(token) for token in tokens]
    return ixs

def decode(encoded, vocab=VOCAB):
    return ('').join([vocab[ix] for ix in encoded if ix not in [0, 1, 2]])
