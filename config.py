
from tkinter import *
import configparser as cp
import go

BLOCKS_NUM = 20                 #残差网络块数         20
FEATURE_NUM = 17                #总特征数             17
RESIDUAL_FILTERS = 256          #残差滤波通道数       256
PLAY_FEATURES = 1               #当前走子特征数       1 
STATE_FEATURES = 2              #当前状态特征数       2
policy_size = 1                   #棋盘边界             1
board_size = 19
net_type = 'fractal'            #网络类型【fractal,resnet,shuffle,fuse】
#除去以上两种特征数，剩下的为历史特征数
running = True                  #控制循环用

mtcs_width = 10                 #蒙特卡洛搜索宽度
mtcs_depth = 10                 #蒙特卡洛搜索深度
mtcs_time = 1                   #蒙特卡洛搜索时间
playouts    = 800
vresign = 0.05                  #投降的胜率阈值       0.05
summary = False                 #是否打开观察记录表
save_dir = 'models'             #模型及权重数据保存起始目录
data_dir = 'processed_data'     #训练的数据保存起始目录
save_name = 'zigo'              #模型及权重数据保存名
trained_model = 'old'           #训练的历史模型和权重数据保存目录
log_dir = 'log'                 #观察记录表数据保存起始目录
learning_rate = 0.1             #初始学习率           0.1
decay_steps = 10000000          #学习率降低的训练步数 200000
decay_rate = 0.1                #学习率降低到的比率 新的=旧的*比率 0.1
momentum = 0.9                  #动量                 0.9
batch_size = 32                 #小批量的大小         32
scope = 'resnet_zero'           #网络域名
last_config = 'config-b6f2'
apps = {}

play_num = 50
test_num = 50
sample_num = 320
test_steps = 5000

maxelo = -90000

def read_cfg(config_file='config'):
    global mtcs_width, mtcs_depth, mtcs_time, vresign, summary, BLOCKS_NUM, FEATURE_NUM
    global learning_rate, decay_steps, decay_rate, momentum, batch_size, scope, save_dir
    global save_name, trained_model, log_dir, RESIDUAL_FILTERS, data_dir, PLAY_FEATURES
    global STATE_FEATURES, last_config, play_num, test_num, sample_num
    global apps, test_steps, net_type, policy_size, board_size, playouts
    config_file = 'config/%s.ini' % (config_file)
    cf = cp.ConfigParser()
    cf.read(config_file)
    if config_file=='config/config.ini':
        last_config = cf.get('PLAY', 'last_config')
    mtcs_width = cf.getint('MTCS', 'width')
    mtcs_depth = cf.getint('MTCS', 'depth')
    mtcs_time = cf.getint('MTCS', 'time')
    playouts = cf.getint('MTCS', 'playouts')
    vresign = cf.getfloat('PLAY', 'vresign')
    board_size = cf.getint('PLAY', 'board_size')
    summary = cf.getboolean('TRAIN', 'summary')
    data_dir = cf.get('TRAIN', 'data_dir')
    save_dir = cf.get('TRAIN', 'save_dir')
    save_name = cf.get('TRAIN', 'save_name')
    trained_model = cf.get('TRAIN', 'trained_model')
    log_dir = cf.get('TRAIN', 'log_dir')
    BLOCKS_NUM = cf.getint('POLICY', 'blocks')
    FEATURE_NUM = cf.getint('POLICY', 'features')
    PLAY_FEATURES = cf.getint('POLICY', 'play_features')
    STATE_FEATURES = cf.getint('POLICY', 'state_features')
    RESIDUAL_FILTERS = cf.getint('POLICY', 'residual_filters')
    net_type = cf.get('POLICY', 'net_type')
    policy_size = cf.getint('POLICY', 'policy_size')
    learning_rate = cf.getfloat('POLICY', 'learning_rate')
    decay_steps = cf.getint('POLICY', 'decay_steps')
    decay_rate = cf.getfloat('POLICY', 'decay_rate')
    momentum = cf.getfloat('POLICY', 'momentum')
    batch_size = cf.getint('POLICY', 'batch_size')
    scope = cf.get('POLICY', 'scope')
    play_num = cf.getint('SELF_TRAIN', 'play_num')
    test_num = cf.getint('SELF_TRAIN', 'test_num')
    sample_num = cf.getint('SELF_TRAIN', 'sample_num')
    test_steps = cf.getint('SELF_TRAIN', 'test_steps')
    apps={}
    namesstr = cf.get("APPS", "names")
    names = namesstr.split(",")
    for name in names:
        cmd = cf.get("APPS", name)
        apps[name] = cmd

def is_validate(content):        #如果你不加上==""的话，你就会发现删不完。总会剩下一个数字
    if content.isdigit() or (content==""):
        return True
    else:
        return False

def isfloat(s):
    m = re.match(r'^[0-9]*\.*[0-9]*$', s)
    if m or (s==""):
        return True
    return False


class ConfigWindow():
    def __init__(self, wdpi, hdpi):
        self.top = Toplevel()
        self.gui(wdpi, hdpi)
        self.top.mainloop()

    def gui(self, wdpi, hdpi):
        global mtcs_width, mtcs_depth, mtcs_time, vresign
        width = int(wdpi * 7)
        height = int(hdpi * 5)
        padx=int(wdpi*0.05)
        pady=int(hdpi*0.05)
        self.top.geometry('%dx%d+%d+%d' % (width, height, wdpi* 3, hdpi*2))
        self.top.resizable(width=True, height=True)
        self.top.title("设置")
        self.vwidth=StringVar()
        self.vdepth=StringVar()
        self.vtime=StringVar()
        self.vvresign=StringVar()
        validate_fun=self.top.register(is_validate)#需要将函数包装一下，必要的
        lbl1=Label(self.top, text='蒙特卡洛搜索宽度：')
        self.txtwidth=Entry(self.top, textvariable = self.vwidth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl2=Label(self.top, text='蒙特卡洛搜索深度：')
        self.txtdepth=Entry(self.top, textvariable = self.vdepth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl3=Label(self.top, text='蒙特卡洛搜索时间：')
        self.txttime=Entry(self.top, textvariable = self.vtime, validate='key', validatecommand=(validate_fun,'%P'))
        lbl4=Label(self.top, text='投降的胜率阈值：')
        self.txtvresign=Entry(self.top, textvariable = self.vvresign, validate='key', validatecommand=(isfloat,'%P'))
        btnOk=Button(self.top, text='确定', command=self.save)
        btnCancel=Button(self.top, text='取消', command=self.cancel)
        lbl1.grid(row = 0, column=0, padx=padx, pady=pady)
        self.txtwidth.grid(row = 0, column=1, padx=padx, pady=pady)
        lbl2.grid(row = 1, column=0, padx=padx, pady=pady)
        self.txtdepth.grid(row = 1, column=1, padx=padx, pady=pady)
        lbl3.grid(row = 2, column=0, padx=padx, pady=pady)
        self.txttime.grid(row = 2, column=1, padx=padx, pady=pady)
        lbl4.grid(row = 3, column=0, padx=padx, pady=pady)
        self.txtvresign.grid(row = 3, column=1, padx=padx, pady=pady)
        btnOk.grid(row = 4, column = 0, padx=padx*4, pady=pady)
        btnCancel.grid(row = 4, column = 1, padx=padx*4, pady=pady)
        self.top.columnconfigure(1, weight = 1)
        self.top.rowconfigure(3, weight = 1)
        self.vwidth.set(str(mtcs_width))
        self.vdepth.set(str(mtcs_depth))
        self.vtime.set(str(mtcs_time))
        self.vvresign.set(str(vresign))

    def save(self, config_file=None):
        global mtcs_width, mtcs_depth, mtcs_time, vresign, last_config
        if config_file:
            last_config = config_file
        else:
            config_file = last_config
        config_file = 'config/%s.ini' % (config_file)
        mtcs_width = int(self.vwidth.get())
        mtcs_depth = int(self.vdepth.get())
        mtcs_time = int(self.vtime.get())
        vresign = float(self.vvresign.get())
        cf = cp.ConfigParser()
        cf.read(config_file)
        cf.set('MTCS', 'width', str(mtcs_width))
        cf.set('MTCS', 'depth', str(mtcs_depth))
        cf.set('MTCS', 'time', str(mtcs_time))
        cf.set('PLAY', 'vresign', str(vresign))
        with open(config_file,"w") as f:
            cf.write(f)
        shutile.copyfile(config_file, 'config/config.ini')
        cf0=cp.ConfigParser()
        cf0.read('config/config.ini')
        cf0.set('PLAY', 'last_config', last_config)
        with open('config/config.ini',"w") as f:
            cf0.write(f)
        self.top.destroy()

    def cancel(self):
        self.top.destroy()
