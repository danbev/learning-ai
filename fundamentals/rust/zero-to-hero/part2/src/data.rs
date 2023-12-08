use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub const WORD_BOUNDARY: char = '.';

pub struct Data {
    words: Vec<String>,
    chars: BTreeSet<char>,
    stoi: HashMap<char, usize>,
    itos: HashMap<usize, char>,
}

impl Data {
    pub fn new(file_name: &str) -> io::Result<Data> {
        let path = Path::new(file_name);
        let words = Data::read(path)?;
        println!("Read {} lines from {}", words.len(), path.display());

        let mut chars: BTreeSet<char> = BTreeSet::new();
        for line in &words {
            for ch in line.chars() {
                chars.insert(ch);
            }
        }
        let mut stoi: HashMap<char, usize> = HashMap::new();
        stoi.insert(WORD_BOUNDARY, 0);
        for (i, ch) in chars.iter().enumerate() {
            stoi.insert(*ch, i + 1);
        }
        let itos: HashMap<usize, char> = stoi.iter().map(|(k, v)| (*v, *k)).collect();

        println!("stoi: {stoi:?}");
        println!("itos: {itos:?}");
        Ok(Self {
            words,
            chars,
            stoi,
            itos,
        })
    }

    fn read(path: &Path) -> io::Result<Vec<String>> {
        let reader = io::BufReader::new(File::open(path)?);
        reader.lines().collect()
    }

    pub fn words(&self) -> &Vec<String> {
        &self.words
    }

    pub fn chars(&self) -> &BTreeSet<char> {
        &self.chars
    }

    pub fn stoi(&self, ch: char) -> usize {
        self.stoi[&ch]
    }
    pub fn itos(&self, i: usize) -> &char {
        &self.itos[&i]
    }
}
