import logging,os 
import ctypes
FOREGROUND_WHITE = 0x0007 
FOREGROUND_BLUE = 0x01 # text color contains blue. 
FOREGROUND_GREEN= 0x02 # text color contains green. 
FOREGROUND_RED = 0x04 # text color contains red. 
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN  
STD_OUTPUT_HANDLE= -11 
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

def set_color(color, handle=std_out_handle):  
    booll = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)  
    return booll

class Logger:  
    def __init__(self, path,clevel = logging.DEBUG, Flevel = logging.DEBUG):   
        self.logger = logging.getLogger(path)   
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)

        # whether the directory of log file path exists
        log_dir = os.path.dirname(path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)

    def war(self,message,color=FOREGROUND_YELLOW):
        set_color(color)
        self.logger.warn(message)  
        set_color(FOREGROUND_WHITE)

    def error(self,message,color=FOREGROUND_RED):
        set_color(color)
        self.logger.error(message)
        set_color(FOREGROUND_WHITE)

    def cri(self,message):
        self.logger.critical(message)

if __name__ =='__main__':  
    logyyx = Logger('yyx.log',logging.WARNING,logging.DEBUG)  
    logyyx.debug('debug')
    logyyx.info('info')
    logyyx.war('warning')
    logyyx.error('error')
    logyyx.cri('critical')