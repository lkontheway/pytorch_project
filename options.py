import argparse
import os


def parse_common_args(parser):
    """
    先用 parse_common_args 添加训练测试共用的一些参数，供 parse_train_args 和 parse_test_args 调用

    :param
        model_type: 模型的名字，配合 model 目录和 model_entry.py 使用；
        data_type：数据集的名字，配合 data 目录和 data_entry.py 使用；
        save_prefix：    训练时：实验的名字，可以备注自己改了那些重要组件，具体的参数，会用于创建保存模型的目录；
                        测试时：测试的名字，可以备注测试时做了哪些配置，会用于创建保存测试结果的目录；
        load_model_path：训练时，作为预训练模型路径，
                        测试时，作为待测模型路径，有的人喜欢传入一个模型名字，再传入一个 epoch ，但其实没啥必要，就算要循环测多个目录，我们也可以写 shell 生成对应的 load_model_path ，而且通常只需要测最后一个 epoch 的模型；
        load_not_strict：我写了一个 load_match_dict 函数（utils/torch_utils.py），允许加载的模型和当前模型的参数不完全匹配，可多可少，如果打开这个选项，就会调用此函数，这样我们就可以修改模型的某个组件，然后用之前的模型来做预训练啦！如果关闭，就会用 torch 原本的加载逻辑，要求比较严格的参数匹配；
        val_list:   训练时可以传入验证集 list ，
                    测试时可以传入测试集 list ；
        gpus：可以配置训练或测试时使用的显卡编号，在多卡训练时需要用到，测试时也可以指定显卡编号，绕开其他正在用的显卡，当然你也可以在命令行里 export CUDA_VISIBLE_DEVICES 这个环境变量来控制

    :return:
    """

    parser.add_argument('--model_type', type=str, default='base_model', help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='base_dataset', help='used in data_entry.py')
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='checkpoints/base_model_pref/0.pth',
                        help='model path for pretrain or test')
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                        help='val list in train, test list path in test')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    return parser


def parse_train_args(parser):
    """

    :param
        lr，momentum, beta, weight-decay:  optmizer 相关参数，在 train.py 中初始化optimizer
        model_dir：模型的存储目录，留空，不用传入，会在 get_train_model_dir 函数中确定这个字段的值，创建对应的目录，填充到 args 中，方便其他模块获得模型路径
        train_list：训练集list路径
        batch_size：训练时的batch size，有人可能会问，为啥测试时不用设置 batch size ？主要是出于测试时的可视化需求，往往测试需要一张一张 forward ，所以我习惯将测试 batch size 为 1
        epochs：模型训练epoch数

    :return:
    """
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    return parser


def parse_test_args(parser):
    """

    :param
        save_viz：控制是否保存可视化结果的开关
        result_dir：可视化结果和测试结果的存储目录，留空，不用传入，会在get_test_result_dir中自动生成，自动创建目录，这个目录通常位于模型路径下，形如checkpoints/model_name/checkpoint_num/val_info_save_prefix

    :return:
    """
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
