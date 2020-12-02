import pyterrier as pt
if not pt.started():
    pt.init(version="snapshot")
    
import tensorflow.compat.v1 as tf
import numpy as np
import collections

from pyterrier.transformer import TransformerBase

from run_reranking import model_fn_builder
from input_parser import input_fn_builder
from bert.modeling import BertConfig
from generate_data import PointwiseInstance
from bert import tokenization



def create_instance_pointwise(tokenizer, max_seq_length, qid, docno, query, doc, label):
  query = tokenization.convert_to_unicode(query)
  doc = tokenization.convert_to_unicode(doc)
  passages = get_passages(doc, 150, 50)
  if len(passages) == 0:
    tf.logging.warn("Passage length is 0 in qid {} docno {}".format(qid, docno))

  query = tokenization.convert_to_bert_input(
    text=query,
    max_seq_length=64,
    tokenizer=tokenizer,
    add_cls=True,
    convert_to_id=False
  )
  passages = [tokenization.convert_to_bert_input(
    text=p,
    max_seq_length=max_seq_length-len(query),
    tokenizer=tokenizer,
    add_cls=False,
    convert_to_id=False
  ) for p in passages]
  instance = PointwiseInstance(
    exampleid="{}-{}".format(qid, docno),
    tokens_a=query,
    tokens_b_list=passages,
    relation_label=label
  )

  return instance

def get_passages(text, plen, overlap):
    """ Modified from https://github.com/AdeDZY/SIGIR19-BERT-IR/blob/master/tools/gen_passages.py
    :param text:
    :param plen:
    :param overlap:
    :return:
    """
    words = text.strip().split(' ')
    s, e = 0, 0
    passages = []
    while s < len(words):
      e = s + plen
      if e >= len(words):
        e = len(words)
      # if the last one is shorter than 'overlap', it is already in the previous passage.
      if len(passages) > 0 and e - s <= overlap:
        break
      p = ' '.join(words[s:e])
      passages.append(p)
      s = s + plen - overlap

    if len(passages) > 8:
      chosen_ids = sorted(random.sample(range(1, len(passages) - 1), 8 - 2))
      chosen_ids = [0] + chosen_ids + [len(passages) - 1]
      passages = [passages[id] for id in chosen_ids]

    return passages



from generate_data import create_int_feature

def convert_tokens_to_ids(vocab, tokens):
    return [vocab[token] for token in tokens]

def write_instance_to_example_files(writer, tokenizer, instance, instance_idx):
    
    def padding_2d(ids_list, num_tokens_per_segment, padding_value=0):
        _len = len(ids_list)
        if padding_value == 0:
            matrix = np.zeros((_len, num_tokens_per_segment), dtype=np.int)
        elif padding_value == 1:
            matrix = np.ones((_len, num_tokens_per_segment), dtype=np.int)
        else:
            raise ValueError("Unsupport padding value")

        for i, _list in enumerate(ids_list):
            matrix[i, :len(_list)] = _list

        return matrix.flatten()

    tokens_a = instance.tokens_a
    tokens_b_list = instance.tokens_b_list
    tokens_a_ids = convert_tokens_to_ids(tokenizer.vocab, tokens_a)
    tokens_b_list = [convert_tokens_to_ids(tokenizer.vocab, p) for p in tokens_b_list]
    label = instance.relation_label
    assert len(tokens_b_list) <= 8
    num_segments = len(tokens_b_list)

    input_ids = [tokens_a_ids + tokens_b_passage_ids for tokens_b_passage_ids in tokens_b_list]
    tokens_a_len = len(tokens_a_ids)  # helpful for segment ids
    input_ids_lens = [len(input_id) for input_id in input_ids]  # helpful for input mask
    input_ids_lens = input_ids_lens + [128] * (8 - len(input_ids_lens))
    input_ids = padding_2d(input_ids, 128, padding_value=0)
    # write to tfrecord
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["tokens_a_len"] = create_int_feature([tokens_a_len])
    features["tokens_ids_lens"] = create_int_feature(input_ids_lens)
    features["num_segments"] = create_int_feature([num_segments])
    features["label"] = create_int_feature([label])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    
    if instance_idx < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens_a: %s" % " ".join(
            [tokenization.printable_text(x) for x in instance.tokens_a]))
        tf.logging.info("tokens_b_list: {}".format(instance.tokens_b_list))

        for feature_name in features.keys():
            feature = features[feature_name]
            values = []
            if feature.int64_list.value:
                values = feature.int64_list.value
            elif feature.float_list.value:
                values = feature.float_list.value
            tf.logging.info(
                "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
            
            
            
def df_to_feature_list(df, tokenizer, writer):
    feature_list = []
    for ind, line in enumerate(df.itertuples()):
        instance = create_instance_pointwise(
                            tokenizer=tokenizer,
                            max_seq_length=512,
                            qid=line.qid,
                            docno=line.docno,
                            query=line.query,
                            doc=line.body,
                            label=0
        )
        
        write_instance_to_example_files(writer, tokenizer, instance, ind)

        
class PARADEPipeline(TransformerBase):
    def __init__(self, aggregation_method):
        self.aggregation_method = aggregation_method #'cls_max',  'cls_avg', 'cls_attn' or 'cls_transformer'

        self.tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt')
        self.writer = tf.python_io.TFRecordWriter("output.tfrecords")

        self.run_config = tf.estimator.tpu.RunConfig(
            cluster=None,
            model_dir=None,
            save_checkpoints_steps=1000,
            keep_checkpoint_max=1,
            tpu_config=tf.estimator.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,
                per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2))

        self.model_fn = model_fn_builder(
            bert_config=BertConfig.from_json_file('bert_models_onMSMARCO/vanilla_bert_tiny_on_MSMARCO/bert_config.json'),
            num_labels=2,
            init_checkpoint='bert_models_onMSMARCO/vanilla_bert_tiny_on_MSMARCO/model.ckpt-1600000',
            learning_rate=5e-5,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=False,
            use_one_hot_embeddings=False,
            aggregation_method=self.aggregation_method,
            pretrained_model='bert',
            from_distilled_student=False)

        self.estimator = tf.estimator.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=32,
            eval_batch_size=32,
            predict_batch_size=32)
        
        
    def transform(self, queries_and_docs):
        def main(_):
            df_to_feature_list(queries_and_docs, self.tokenizer, self.writer) #writes the dataframe to the filepath specified in the writer declaration

            eval_input_fn = input_fn_builder(
                dataset_path="output.tfrecords",
                max_num_segments_perdoc=8,
                max_seq_length=128,
                is_training=False)
        
            result = self.estimator.predict(input_fn=eval_input_fn, yield_single_examples=True)
            
            print(list(result))
        
        tf.app.run(main=main)
        
        
        
        
        
import pandas as pd
q = "chemical reactions"
doc1 = "professor proton demonstrated the chemical reaction"
doc2 = "chemical brothers is great techno music"

df = pd.DataFrame([["q1", q, "doc1", doc1]], columns=["qid", "query", "docno", "body"])





pipeline = PARADEPipeline(aggregation_method='cls_max')

#vaswani  = pt.datasets.get_dataset("vaswani")
#vaswani.get_corpus()


pipeline(df)