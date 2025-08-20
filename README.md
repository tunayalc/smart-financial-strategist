# Akıllı Finansal Stratejist

Bu proje, standart finansal tavsiye motorlarının ötesine geçerek, her bireyin benzersiz hedeflerini, hikayesini ve risk algısını anlayan bir yapay zeka sistemidir. Sistem, bu kişisel bilgileri tarihsel piyasa verileri ve kanıtlanmış finansal modellerle birleştirerek, tamamen kişiye özel, her adımının gerekçelendirildiği ve eyleme geçirilebilir bir yatırım stratejisi üretir.

Projenin temel vizyonu, bir "hesap makinesi" değil, kullanıcısının finansal yolculuğunda güvenilir bir **"strateji ortağı"** olmaktır.

---

## Çalışma Prensibi ve Mimari

Sistem, üç uzman modülün sinerjisi üzerine kurulmuştur. Her modül, yapay zeka ajanının belirli bir soruyu yanıtlaması için tasarlanmıştır:

### 1. Analitik Çekirdek (The Quant): "NE Yapılmalı?"
Bu modül, projenin tarafsız ve veri odaklı beynidir. Geçmiş piyasa verilerini (BIST, S&P 500, Emtialar, Kripto) analiz ederek Nobel ödüllü **Modern Portföy Teorisi (MPT)** prensiplerini uygular. Monte Carlo simülasyonları ile binlerce potansiyel portföyü değerlendirir ve kullanıcının risk profiline göre matematiksel olarak en verimli varlık dağılımını (`Efficient Frontier`) belirler.

### 2. Bilgi Havuzu (The Librarian - RAG): "NEDEN Yapılmalı?"
Bu modül, sistemin bilgeliği ve hafızasıdır. Yatırım stratejileri, varlık sınıflarının analizleri ve risk yönetimi üzerine seçilmiş uzman makalelerinden oluşan bir bilgi tabanı, bir vektör veritabanında (`ChromaDB`) saklanır. **Retrieval-Augmented Generation (RAG)** mimarisi sayesinde, Analitik Çekirdek'in önerdiği her bir varlık veya stratejinin arkasındaki mantığı ve gerekçeyi bu bilgi havuzundan kanıta dayalı olarak çeker.

### 3. Üretken Ajan (The Strategist - LLM): "Sentez ve İletişim"
Sistemin kalbi ve sesi olan bu katman, `Ollama` üzerinde çalışan **Llama 3.1** modeli ve `LangChain` ajan framework'ü ile hayata geçirilmiştir. Ajanın görevi, bir orkestra şefi gibi diğer iki modülü yönetmektir:
1.  Kullanıcının hedefini doğal dilde anlar.
2.  **Analitik Çekirdek**'ten gelen sayısal portföy önerisini alır.
3.  **Bilgi Havuzu**'nu kullanarak bu önerinin arkasındaki mantığı ve gerekçeleri toplar.
4.  Tüm bu yapılandırılmış (sayısal veri) ve yapılandırılmamış (metin) bilgiyi sentezleyerek, kullanıcıya özel, akıcı ve motive edici bir dille yazılmış bütünsel bir finansal yol haritası üretir.

---

## Kullanılan Teknolojiler ve Ortam

Bu proje, modern yapay zeka ve yazılım geliştirme araçlarını bir araya getiren, tamamen Dockerize edilmiş bir ortamda geliştirilmiştir.

-   **Altyapı ve Konteyner Yönetimi:** Docker, Docker Compose, NVIDIA CUDA Desteği
-   **Yapay Zeka ve Makine Öğrenmesi:** Ollama, Llama 3.1, LangChain, Sentence-Transformers
-   **Finansal Modelleme ve Veri İşleme:** PyPortfolioOpt, yfinance, Pandas, NumPy
-   **Backend ve Veritabanı:** Python 3.11, ChromaDB
-   **Frontend:** Streamlit

---

## Proje Dosyaları ve Görevleri

Projenin her bir parçasının ne işe yaradığının dökümü aşağıdadır.

### Ana Dizin Dosyaları

-   **`docker-compose.yml`**
    -   **Görevi:** Projenin Docker servislerini ve bu servislerin birbirleriyle olan ilişkilerini tanımlar. `stratejist-app` adında tek bir servis oluşturur ve bu servisin ağ ayarlarını, port yönlendirmelerini, volume bağlantılarını ve GPU erişim yetkilerini yapılandırır. Projenin altyapısal bel kemiğidir.

-   **`Dockerfile`**
    -   **Görevi:** `stratejist-app` konteynerinin nasıl inşa edileceğini adım adım tarif eden bir dosyadır. Temel Python imajını belirler, gerekli kütüphaneleri `requirements.txt`'den yükler, proje kodlarını konteynerin içine kopyalar ve son olarak konteyner başladığında çalıştırılacak olan `streamlit` komutunu tanımlar.

-   **`requirements.txt`**
    -   **Görevi:** Projenin çalışması için gerekli olan tüm Python kütüphanelerini ve versiyonlarını listeleyen bir metin dosyasıdır. `Dockerfile`, bu dosyayı kullanarak projenin tüm bağılılıklarını otomatik olarak kurar.

-   **`.gitignore`**
    -   **Görevi:** Git versiyon kontrol sistemine, hangi dosya ve klasörlerin takip edilmemesi gerektiğini söyler. Otomatik oluşturulan klasörler (`chroma_db`, `__pycache__`) veya gizli bilgiler içeren dosyalar (`.env` gibi) bu dosya sayesinde repository'e gönderilmez, böylece temiz ve güvenli bir kod tabanı korunur.

-   **`README.md`**
    -   **Görevi:** Bu dosya. Projenin amacını, mimarisini, kullanılan teknolojileri ve dosya yapısını açıklayan ana bilgilendirme dokümanıdır. Projeyi ziyaret edenler için bir başlangıç noktası ve rehber görevi görür.

### `src/` Klasörü ve İçindeki Dosyalar

Bu klasör, uygulamanin tüm mantıksal ve fonksiyonel kodlarını barındırır.

-   **`src/app.py`**
    -   **Görevi:** Projenin kullanıcıya bakan yüzüdür. `Streamlit` kütüphanesini kullanarak interaktif bir web arayüzü oluşturur. Kullanıcıdan finansal hedefini alır, yapay zeka ajanını bu girdiyle tetikler ve ajan tarafından üretilen nihai raporu ekranda formatlı bir şekilde gösterir.

-   **`src/agent_tools.py`**
    -   **Görevi:** Projenin modüler yapısının merkezindeki entegrasyon dosyasıdır. Diğer modüllerde (`quant_core.py`, `rag_core.py`) bulunan karmaşık fonksiyonları, `LangChain` ajanının anlayabileceği ve kullanabileceği standart "araç" (tool) formatına dönüştürür. Bu sayede LLM, "portföy hesapla" veya "gerekçe bul" gibi komutlarla bu fonksiyonları çağırabilir.

-   **`src/quant_core.py`**
    -   **Görevi:** Projenin "Quant" (sayısal analist) modülüdür. `yfinance` aracılığıyla finansal verileri çeker, `PyPortfolioOpt` kütüphanesini kullanarak Modern Portföy Teorisi'ne dayalı risk-getiri optimizasyonları yapar ve sonuç olarak sayısal bir varlık dağılımı önerisi sunar.

-   **`src/rag_core.py`**
    -   **Görevi:** Projenin "Librarian" (kütüphaneci) modülüdür. `knowledge_base` klasöründeki metin belgelerini okur, bunları `Sentence-Transformers` ile anlamsal vektörlere dönüştürür ve `ChromaDB` üzerinde aranabilir bir vektör veritabanı oluşturur. Ajanın sorduğu konularla ilgili en alakalı metin parçalarını bularak geri döndürür.

-   **`src/knowledge_base/` (Klasör)**
    -   **Görevi:** RAG sisteminin bilgi kaynağını içeren klasördür. İçindeki her dosya, belirli bir finansal konsepti RAG sistemine öğretir. Sistem, "Neden?" sorularına cevap verirken bu dosyalardaki metinlerden faydalanır.
        -   **`altin.txt`**: Altının ekonomik belirsizlik dönemlerinde neden 'güvenli liman' olarak kabul edildiğini ve portföydeki dengeleyici rolünü açıklar.
        -   **`cesitlendirme.txt`**: Yatırım riskini dağıtmanın temel prensibi olan portföy çeşitlendirmesinin önemini ve nasıl çalıştığını anlatır.
        -   **`hisse_senetleri.txt`**: Hisse senetlerinin ne olduğunu, uzun vadeli büyüme potansiyelini ve taşıdığı riskleri özetler.
