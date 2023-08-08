import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,BertTokenizer, BertForSequenceClassification

def load_model():
    #model = DistilBertForSequenceClassification.from_pretrained("models", cache_dir="./cache")  # Model dosyaları ./cache klasörüne indirilecek
    model = DistilBertForSequenceClassification.from_pretrained("results/checkpoint-60000", cache_dir="./cache")
    tokenizer = DistilBertTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased", cache_dir="./cache")  # Tokenizer dosyaları ./cache klasörüne indirilecek
    return model, tokenizer



def classify_text(text, model, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    return predicted_class

model, tokenizer = load_model()

sample_texts = [
    "müşteri hizmetleri çok yavaş cevap veriyor",
    "Kredi başvurusu yaparken görevli faiz oranları ve diğer masraflar konusunda eksik bilgi verdi şeffaf olunmalı",
    "hesabımdan düzenli olarak para kayboluyor gibi görünüyor hesap aktivitelerimi incelenmesini talep ediyorum",
    "atm para çekme işlemi başarısız oldu.",
    "kredi kartımın borcu hatalı olarak gösteriliyor.",
    "internet bankacılığı sisteminiz yavaş ve karmaşık  işlem yapmak neredeyse imkansız",
    "kredi kartı borcumu zamanında ödediğim halde faiz uygulandı  bu durumu düzeltmenizi istiyorum",
    "banka şubesindeki görevlileriniz son derece ilgisiz ve yardımcı olmaktan uzak  sorunlarımı çözmek için saatlerimi harcadım",
    "bankanızın çağrı merkezine defalarca ulaşmaya çalıştım ancak sürekli olarak beklemeye yönlendirildim ve kimseye bağlanamadım.",
    "hesabımdan yanlışlıkla yapılan bir para transferi sonrasında paranın geri iadesi için neden bu kadar uzun süre bekletiliyorum?",
    "Garanti Bankasi kredi kartimdan benim onayim olmadan web pos 1 market Istanbul aciklamali 4958.00 TL'lik bir alisveris yapilmis"
]

for text in sample_texts:
    predicted_class = classify_text(text, model, tokenizer)
    label2id = {0: "müşteri hizmetleri",
                1: "kredi başvurusu",
                2: "mobil uygulama",
                3: "ücretlendirme",
                4: "atm",
                5: "kredi kartı",
                6: "internet bankacılığı",
                7: "faiz işlem",
                8: "şube hizmeti",
                9: "çağrı merkezi",
                10: "para transferi",
                11: "pos", }

    predicted_category = label2id[predicted_class]
    print(f"Metin: {text}")
    print(f"Tahmin: {predicted_category}\n")