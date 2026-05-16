import os
import numpy as np
import librosa
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

DIALECTS = [
    "صنعاء","عدن","شبوة","حضرموت الساحل","ذمار",
    "الحديدة","تعز","الضالع","إب","البيضاء","المحويت",
]
NUM_CLASSES = len(DIALECTS)
IDX2DIALECT = {i: d for i, d in enumerate(DIALECTS)}

DIALECT_RU = {
    "صنعاء":"Сана","عدن":"Аден","شبوة":"Шабва",
    "حضرموت الساحل":"Хадрамаут","ذمار":"Зимар",
    "الحديدة":"Ходейда","تعز":"Таиз","الضالع":"Эд-Далеа",
    "إب":"Ибб","البيضاء":"Эль-Байда","المحويت":"Эль-Махвит",
}

DIALECT_REGION = {
    "صنعاء":{"ru":"Северо-центральный, столица","ar":"المنطقة الشمالية الوسطى"},
    "عدن":{"ru":"Юг Йемена, портовый город","ar":"جنوب اليمن"},
    "شبوة":{"ru":"Центральный регион","ar":"المنطقة الوسطى"},
    "حضرموت الساحل":{"ru":"Восточное побережье","ar":"الساحل الشرقي"},
    "ذمار":{"ru":"Горный центр","ar":"المركز الجبلي"},
    "الحديدة":{"ru":"Западное побережье","ar":"الساحل الغربي"},
    "تعز":{"ru":"Центрально-западный","ar":"المنطقة الوسطى الغربية"},
    "الضالع":{"ru":"Центрально-южный","ar":"المنطقة الوسطى الجنوبية"},
    "إب":{"ru":"Юго-западный","ar":"المنطقة الجنوبية الغربية"},
    "البيضاء":{"ru":"Центральный регион","ar":"المنطقة الوسطى"},
    "المحويت":{"ru":"Северо-западный","ar":"المنطقة الشمالية الغربية"},
}

SAMPLE_RATE=16000; N_SAMPLES=128000; N_FFT=2048
HOP_LENGTH=160; WIN_LENGTH=400; N_MELS=128
FMIN=0; FMAX=8000

def load_audio(path):
    y,_=librosa.load(path,sr=SAMPLE_RATE,mono=True)
    y,_=librosa.effects.trim(y,top_db=60)
    y=librosa.util.normalize(y)
    if len(y)<N_SAMPLES: y=np.pad(y,(0,N_SAMPLES-len(y)))
    else: y=y[:N_SAMPLES]
    return y.astype(np.float32)

def extract_mel(y):
    mel=librosa.feature.melspectrogram(y=y,sr=SAMPLE_RATE,
        n_fft=N_FFT,hop_length=HOP_LENGTH,win_length=WIN_LENGTH,
        n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    return librosa.power_to_db(mel,ref=np.max).astype(np.float32)

class DialectPredictor:
    _instance=None
    def __new__(cls,*a,**k):
        if cls._instance is None:
            cls._instance=super().__new__(cls)
            cls._instance._initialized=False
        return cls._instance
    def __init__(self,model_name="lstm"):
        if self._initialized: return
        self.model_name=model_name
        self._load_model()
        self._initialized=True
    def _load_model(self):
        h5=MODELS_DIR/f"{self.model_name}_best.h5"
        tflite=MODELS_DIR/f"{self.model_name}_best.tflite"
        if tflite.exists():
            import tensorflow as tf
            self.backend="tflite"
            self.interp=tf.lite.Interpreter(model_path=str(tflite))
            self.interp.allocate_tensors()
            self.inp=self.interp.get_input_details()
            self.out=self.interp.get_output_details()
        elif h5.exists():
            import tensorflow as tf
            self.backend="keras"
            self.model=tf.keras.models.load_model(str(h5),compile=False)
        else:
            raise FileNotFoundError(f"Model not found in {MODELS_DIR}")
    def _probs(self,x):
        if self.backend=="keras":
            return self.model.predict(x,verbose=0)[0]
        import tensorflow as tf
        self.interp.resize_tensor_input(self.inp[0]["index"],x.shape)
        self.interp.allocate_tensors()
        self.interp.set_tensor(self.inp[0]["index"],x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out[0]["index"])[0]
    def predict(self,audio_path,top_n=3):
        y=load_audio(audio_path)
        mel=extract_mel(y)
        x=mel[np.newaxis,...,np.newaxis].astype(np.float32)
        probs=self._probs(x)
        top_idx=probs.argsort()[::-1][:top_n]
        top_list=[]
        for i in top_idx:
            d=IDX2DIALECT[int(i)]
            top_list.append({"index":int(i),"dialect_ar":d,
                "dialect_ru":DIALECT_RU[d],"region_ru":DIALECT_REGION[d]["ru"],
                "region_ar":DIALECT_REGION[d]["ar"],
                "probability":float(probs[i]),
                "confidence_pct":round(float(probs[i])*100,2)})
        b=top_list[0]
        return {"model_used":self.model_name,"backend":self.backend,
            "top_dialect_ar":b["dialect_ar"],"top_dialect_ru":b["dialect_ru"],
            "top_region_ru":b["region_ru"],"top_confidence_pct":b["confidence_pct"],
            "top_n":top_list}

_predictor=None
def get_predictor(model_name="lstm"):
    global _predictor
    if _predictor is None:
        _predictor=DialectPredictor(model_name)
    return _predictor
