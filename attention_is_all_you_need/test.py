from tokenizer import SimpleTokenizer

tok = SimpleTokenizer.load("out/tokenizer.json")
ids = [tok.stoi.get("The",0), tok.stoi.get("history",0), tok.stoi.get("of",0), tok.stoi.get("artificial",0)]
print("ids ->", ids)
print("decoded:", tok.decode(ids))
