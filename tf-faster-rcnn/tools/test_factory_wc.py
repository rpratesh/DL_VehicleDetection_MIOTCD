import _init_paths
from model.train_val import get_training_roidb
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb

if __name__ == "__main__":

  print "it's here"
  imdb = get_imdb('voc_2007_trainval')
  print('Loaded dataset `{:s}` for training'.format(imdb.name))

  # imdb = get_imdb('mio_tcd_loc_train')
  # print('Loaded dataset `{:s}` for training'.format(imdb.name))

  cfg.TRAIN.PROPOSAL_METHOD = 'gt'

  # TODO: may need to look more closely at roi_data_layer.roidb
  imdb, roidb = combined_roidb('deep_fashion_general_train')
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, None)
  print('Output will be saved to `{:s}`'.format(output_dir))