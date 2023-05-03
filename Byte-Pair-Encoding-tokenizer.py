from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

trainer = trainers.BpeTrainer(vocab_size=30000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tokenizer.train(["lawiki-latest-pages-articles-multistream.xml"], trainer)

tokenizer.save("latin_tokenizer.json")

