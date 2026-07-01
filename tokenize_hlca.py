from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "sample_id": "sample_id",
        "technology": "technology",
        "disease": "disease",
        "cell_type": "cell_type",
    }
)
tk.tokenize_data(
    data_directory="data/geneformer_input/hlca",
    output_directory="tokenized",
    output_prefix="hlca",
    file_format="h5ad",
)
print("tokenized -> tokenized/hlca.dataset")
