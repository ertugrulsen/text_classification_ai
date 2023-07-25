Proje Özeti: Metin Sınıflandırma ile Şikayet Analizi

Bu proje, bir bankanın şikayet platformuna gelen metinleri otomatik olarak sınıflandırmak için yapay zeka tabanlı bir çözüm sunmaktadır. Şikayetler, kullanıcılar tarafından kredi kartı, kredi, ATM ve müşteri hizmetleri gibi çeşitli kategorilere ayrılarak işlenir. Metin sınıflandırma modeli olarak Hugging Face tarafından sağlanan DistilBERT modeli kullanılmıştır.

Projenin Ana Adımları:

Veri Toplama: Banka şikayet platformundan alınan metin verileri toplanır ve kategorilere göre etiketlenir.
Veri Ön İşleme: Metin verileri temizlenir, önişlemeye tabi tutulur ve DistilBERT modelinin anlayabileceği formata dönüştürülür.
Model Eğitimi: DistilBERT modeli, metin verilerinin kategorilerini öğrenmek için eğitilir.
Model Değerlendirmesi: Eğitilen model, test veri kümesi üzerinde değerlendirilir ve performansı ölçülür.
API Entegrasyonu: Eğitilen model, Flask framework kullanılarak API olarak sunulur. Kullanıcılar, API aracılığıyla metinleri otomatik olarak sınıflandırabilirler.
Projenin Amacı:
Bu proje, banka şikayet platformuna gelen yüksek miktardaki metni etkin bir şekilde sınıflandırmak ve analiz etmek için geliştirilmiştir. Otomatik metin sınıflandırma, şikayetlerin hızlı bir şekilde ilgili birimlere yönlendirilmesini sağlayarak müşteri memnuniyetini artırır ve şikayet yönetim süreçlerini optimize eder.

Bu proje, metin sınıflandırma ve doğal dil işleme alanında kullanılabilecek geniş bir veri kümesi ve eğitilmiş modelle birlikte sunulmaktadır. Ayrıca, modelin API entegrasyonu sayesinde kullanıcılar, kendi metin verilerini sınıflandırma için kolayca kullanabilirler.

Not: Bu özet, proje detaylarına ve amaçlarına genel bir bakış sunmak için hazırlanmıştır. Projenin kapsamı ve içeriği daha ayrıntılı bir açıklama gerektirebilir.