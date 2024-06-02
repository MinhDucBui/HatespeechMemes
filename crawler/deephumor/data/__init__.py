__all__ = [
    'SPECIAL_TOKENS', 'Vocab', 'build_vocab', 'build_vocab_from_file',
    'Tokenizer', 'WordPunctTokenizer', 'CharTokenizer',
    'MemeDataset', 'pad_collate'
]

SPECIAL_TOKENS = {
    'PAD': '<pad>',
    'UNK': '<unk>',
    'BOS': '<bos>',
    'EOS': '<eos>',
    'SEP': '<sep>',
    'EMPTY': '<emp>',
}