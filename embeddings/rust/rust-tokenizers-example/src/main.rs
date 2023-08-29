use cached_path::{Cache, Error, ProgressBar};
use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::BaseVocab;
use std::path::PathBuf;

fn main() {
    let vocab_path = download_file_to_cache(
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    )
    .unwrap();

    let strip_accents = false;
    let lower_case = false;
    let tokenizer: BaseTokenizer<BaseVocab> =
        BaseTokenizer::from_file(&vocab_path, lower_case, strip_accents).unwrap();

    let text_1 = "Red Hat AMQ Broker";
    let encoded = tokenizer.encode(text_1, None, 5, &TruncationStrategy::LongestFirst, 2);

    println!("Encoded: {encoded:?}");
}

fn download_file_to_cache(src: &str) -> Result<PathBuf, Error> {
    let mut cache_dir = dirs::home_dir().unwrap();
    cache_dir.push(".cache");
    cache_dir.push(".rust_tokenizers");

    let cached_path = Cache::builder()
        .dir(cache_dir)
        .progress_bar(Some(ProgressBar::Light))
        .build()?
        .cached_path(src)?;
    Ok(cached_path)
}
