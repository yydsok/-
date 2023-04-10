import pandas as pd
import torchtext
from sklearn.model_selection import train_test_split
from torchtext.legacy import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle


def read_data(opt):
    """
    obtain the src_data and trg_data,
    the data form likes ['I feel lost.', 'hello', 'I feel weak.']
    """
    # 英文-》法语，src：英语，trg:法语
    # if opt.src_data is not None:
    #     try:
    #         opt.src_data = open(opt.src_data, encoding='utf-8').read().strip().split('\n')
    #     except:
    #         print("error: '" + opt.src_data + "' file not found")
    #         quit()
    #
    # if opt.trg_data is not None:
    #     try:
    #         opt.trg_data = open(opt.trg_data, encoding='utf-8').read().strip().split('\n')
    #     except:
    #         print("error: '" + opt.trg_data + "' file not found")
    #         quit()
    if opt.src_train is not None:
        try:
            opt.src_train = open(opt.src_train, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_train + "' file not found")
            quit()
    if opt.trg_train is not None:
        try:
            opt.trg_train = open(opt.trg_train, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_train + "' file not found")
            quit()
    if opt.src_test is not None:
        try:
            opt.src_test = open(opt.src_test, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_test + "' file not found")
            quit()
    if opt.trg_test is not None:
        try:
            opt.trg_test = open(opt.trg_test, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_test + "' file not found")
            quit()
    if opt.src_val is not None:
        try:
            opt.src_val = open(opt.src_val, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_val + "' file not found")
            quit()
    if opt.trg_val is not None:
        try:
            opt.trg_val = open(opt.trg_val, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_val + "' file not found")
            quit()


def create_fields(opt):
    """
    return(SRC, TRG)
    """
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'en_core_web_sm', 'fr_core_news_md']

    # if opt.src_lang not in spacy_langs:
    #     print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)
    # if opt.trg_lang not in spacy_langs:
    #     print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")

    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    # 将文本转化为tensor
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)


    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
            """
            在Python中长久的保存字符串、列表、字典等数据，方便以后使用，而不是简单的放入内存中。
            这个时候Pickle模块就派上用场了，它可以将对象转换为一种可以传输或存储的格式。
            """
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG):
    """return
    [[ batch1 ]
     [ batch2 ]
     ...
    ]
    """

    print("creating dataset and iterator... ")

    # raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    # df = pd.DataFrame(raw_data, columns=["src", "trg"])
    train_raw_data = {'src': [line for line in opt.src_train], 'trg': [line for line in opt.trg_train]}
    # print("train_raw_data.size",train_raw_data.__sizeof__())
    train_df = pd.DataFrame(train_raw_data, columns=["src", "trg"])
    test_raw_data = {'src': [line for line in opt.src_test], 'trg': [line for line in opt.trg_test]}
    test_df = pd.DataFrame(test_raw_data, columns=["src", "trg"])
    val_raw_data = {'src': [line for line in opt.src_val], 'trg': [line for line in opt.trg_val]}
    val_df = pd.DataFrame(val_raw_data, columns=["src", "trg"])


    # remove the lenth larger than max_strlen(default = 80)
    # mask return a Series of true or false
    # mask = (df['src'].str.count('') < opt.max_strlen+2) & (df['trg'].str.count('') < opt.max_strlen+2)
    # df = df.loc[mask]
    train_mask = (train_df['src'].str.count('') < opt.max_strlen + 2) & (
                train_df['trg'].str.count('') < opt.max_strlen + 2)
    train_df = train_df.loc[train_mask]
    test_mask = (test_df['src'].str.count('') < opt.max_strlen + 2) & (
                test_df['trg'].str.count('') < opt.max_strlen + 2)
    test_df = test_df.loc[test_mask]
    val_mask = (val_df['src'].str.count('') < opt.max_strlen + 2) & (val_df['trg'].str.count('') < opt.max_strlen + 2)
    val_df = val_df.loc[val_mask]

    # print(df['src'].str.count(''))

    # 8:1:1划分
    tra = train_df
    test = test_df
    val = val_df
    # tra, val = train_test_split(train_df, test_size=0.2)
    # val, test = train_test_split(val_temp, test_size=0.5)

    tra.to_csv("translate_transformer_temp_train.csv", index=False)
    val.to_csv("translate_transformer_temp_val.csv", index=False)
    test.to_csv("translate_transformer_temp_test.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]

    train = data.TabularDataset.splits(path='.', train='translate_transformer_temp_train.csv',
                                       format='csv', fields=data_fields)[0]
    # print("train", train)
    val, test = data.TabularDataset.splits(path='.', train='translate_transformer_temp_val.csv',
                                           validation='translate_transformer_temp_test.csv', format='csv',
                                           fields=data_fields)
    # train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    # val, test = data.TabularDataset.splits(path='.', train='translate_transformer_temp_val.csv',
    #                                        validation='translate_transformer_temp_test.csv', format='csv',
    #                                        fields=data_fields)
    # return [{'src':['token1','token2'...], 'trg':['token1','token2'...]},
    #        {'src':['token1','token2'...], 'trg':['token1','token2'...]},
    #        {'src':['token1','token2'...], 'trg':['token1','token2'...]}]
    # print((train.__getitem__(0).__dict__()))

    # 构造batch
    # return:
    # [{'src': ['1', '2'], 'trg': ['0', '1']}, {'src': ['2', '3'], 'trg': ['1', '2']}(一个batch)]
    # [一个batch]
    # .....
    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    # 测试集的batchsize有特殊的要求吗？
    test_iter = MyIterator(test, batch_size=opt.batchsize, device=opt.device,
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=False, shuffle=False)
    val_iter = MyIterator(val, batch_size=opt.batchsize, device=opt.device,
                          repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                          batch_size_fn=batch_size_fn, train=False, shuffle=False)

    os.remove('translate_transformer_temp_train.csv')
    os.remove('translate_transformer_temp_test.csv')
    os.remove('translate_transformer_temp_val.csv')

    # 初始化
    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']  # stoi：把字符映射成数字，itos：把数字映射成字符
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)  # batch的个数
    opt.test_len = get_len(test_iter)
    opt.val_len = get_len(val_iter)

    return train_iter, test_iter, val_iter


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i
