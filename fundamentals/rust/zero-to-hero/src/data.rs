use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub struct Data {
    file_name: String,
    words: Vec<String>,
    chars: BTreeSet<char>,
    stoi: HashMap<char, usize>,
    itos: HashMap<usize, char>,
}

impl Data {
    pub fn new(file_name: &str) -> Data {
        let path = Path::new(file_name);
        let file = File::open(&path).unwrap();
        let words: Vec<String> = io::BufReader::new(file)
            .lines()
            .map(|l| l.unwrap())
            .collect();
        println!("Read {} lines from {}", words.len(), path.display());

        let mut chars: BTreeSet<char> = BTreeSet::new();
        for line in &words {
            for ch in line.chars() {
                chars.insert(ch);
            }
        }
        let mut stoi: HashMap<char, usize> = HashMap::new();
        stoi.insert('.', 0);
        for (i, ch) in chars.iter().enumerate() {
            stoi.insert(*ch, i + 1);
        }
        let itos: HashMap<usize, char> = stoi.iter().map(|(k, v)| (*v, *k)).collect();
        Self {
            file_name: file_name.to_string(),
            words,
            chars,
            stoi,
            itos,
        }
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn words(&self) -> &Vec<String> {
        &self.words
    }

    pub fn chars(&self) -> &BTreeSet<char> {
        &self.chars
    }

    pub fn stio(&self, ch: char) -> &usize {
        &self.stoi[&ch]
    }
    pub fn itos(&self, i: usize) -> &char {
        &self.itos[&i]
    }
}
