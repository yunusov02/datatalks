# Model Evaluation — To'liq Tushuntirish

Bu fayl `evaluation.ipynb` notebookdagi accuracy'dan keyingi barcha qadamlarni chuqur tushuntiradi: **nima qilinayapti, nega qilinayapti, shundan qanday xulosa/qaror chiqdi, va agar boshqacha qilinganda nima bo'lardi.**

---

## 1. Accuracy va Dummy Model muammosi

### Nima qilindi
Threshold'ni (0 dan 1 gacha, 0.05 qadam bilan) o'zgartirib, har birida accuracy hisoblandi:

```python
for t in tresholds:
    ac = accuracy_score(y_val, y_pred > t)
```

Natija: eng yaxshi accuracy **t=0.5** da — **80.27%**.

### Nega bu qilindi
Model `predict_proba` orqali ehtimollik (0-1 oralig'ida son) chiqaradi, lekin "churn qiladimi yo'qmi" degan **binary qaror** kerak. Shu qarorni qayerdan (qaysi ehtimollikdan boshlab "churn" deb hisoblaymiz) qilishni bilish uchun turli tresholdlarni sinab ko'rish kerak edi.

### Qanday qarorga keldik
t=0.5 threshold eng yaxshi accuracy bergani uchun, dastlab shu tanlangan. Lekin bitta muhim narsa payqaldi:

- **t=1.0** da ham accuracy **73.1%** chiqdi.
- t=1.0 degani — model **hech kimni** "churn qiladi" demaydi (chunki hech qanday ehtimollik 1dan katta bo'lmaydi).
- Bu — eng oddiy, "aqlsiz" model: **dummy model** (har doim ko'pchilik klassni, ya'ni "no churn"ni bashorat qiladi).

### Nega bu muammo
Bizning haqiqiy modelimiz (80.27%) dummy modeldan (73.1%) atigi **~7 foiz punkt** yaxshiroq. Bu juda kichik farq — model deyarli hech narsa "o'rganmagan" bo'lishi ham mumkin edi, lekin accuracy buni yashiradi.

**Sabab: ma'lumotlar disbalans.** `churn=1` (ketganlar) — atigi ~26.9%, `churn=0` (qolganlar) — ~73.1%. Disbalans ma'lumotda har qanday "hammani ko'pchilik klass deb ayt" degan ahmoq model ham yuqori accuracy oladi, chunki ko'pchilik allaqachon shu klassda.

### Agar buni tekshirmaganimizda nima bo'lardi
Accuracy=80% raqamiga qarab "model juda yaxshi ishlayapti" deb xulosa qilinardi, holbuki u dummy modeldan unchalik farq qilmaydi. Bu **noto'g'ri ishonch** yaratadi — production'da model haqiqatda churn qiluvchilarni deyarli topa olmasligi mumkin, lekin accuracy raqami buni ko'rsatmaydi.

### Xulosa
**Accuracy yagona metrika sifatida disbalans ma'lumotda ishlatib bo'lmaydi.** Shuning uchun keyingi qadamlarga (Confusion Matrix, Precision/Recall, ROC) o'tildi.

---

## 2. Confusion Matrix

### Nima qilindi
```python
tp = ((y_pred > t) & (y_val == 1)).sum()
tn = ((y_pred < t) & (y_val == 0)).sum()
fp = ((y_pred > t) & (y_val == 0)).sum()
fn = ((y_pred < t) & (y_val == 1)).sum()
```
Natija (t=0.5 da):
```
[[928, 102],
 [176, 203]]
```

### Bu nima degani
| | Bashorat: No churn | Bashorat: Churn |
|---|---|---|
| **Haqiqat: No churn** | TN = 928 | FP = 102 |
| **Haqiqat: Churn** | FN = 176 | TP = 203 |

- **TP=203** — chindan ketganlardan to'g'ri topilganlar.
- **TN=928** — chindan qolganlardan to'g'ri topilganlar.
- **FP=102** — qolib ketmoqchi bo'lmagan mijozga bekorga "ketadi" deb signal berildi (masalan bekorga chegirma taklif qilinadi — kichik zarar).
- **FN=176** — chindan ketayotgan 176 ta mijoz **sezilmay qoldi** — hech qanday chora ko'rilmaydi, mijoz yo'qotiladi (katta zarar).

### Nega bu hisoblandi
Accuracy bitta son bilan "necha foiz to'g'ri" deydi, lekin **qaysi turdagi xato** ko'proq ekanligini yashiradi. Bu yerda FN (176) — biznes uchun eng qimmat xato, chunki ketayotgan mijozni yo'qotib qo'yasiz va hech narsa qila olmaysiz.

### Qanday qarorga keldik
Bu jadval accuracy'ning "80% to'g'ri" degan tasdig'i ortida **176 ta muhim mijozni yo'qotish xavfi** borligini ko'rsatdi. Shuning uchun keyingi qadamda aynan shu FP/FN nisbatlarini o'lchaydigan metrikalar (Precision, Recall) hisoblandi.

### Agar bu qilinmaganda
Model qanday xato qilayotganini bilmay qolar edik — masalan model ko'proq FN qilyaptimi yoki FP qilyaptimi, buni bilmasdan turib qaysi threshold biznes uchun to'g'riroq ekanini tanlab bo'lmaydi.

---

## 3. Precision, Recall, F1

### Nima qilindi
```python
p = tp / (tp + fp)   # = 0.665
r = tp / (tp + fn)   # = 0.536
f1 = 2 * p * r / (p + r)   # = 0.594
```

### Bu nima
- **Precision** — "churn" deb bashorat qilinganlardan qanchasi haqiqatan churn qildi. Past precision = ko'p bekorga signal (FP ko'p).
- **Recall** — haqiqiy churn qiluvchilardan qanchasi topildi. Past recall = ko'p mijoz sezilmay qoldi (FN ko'p).
- **F1** — ikkalasining muvozanatli o'rtachasi (harmonic mean) — ikkalasi ham muhim bo'lganda bitta son bilan solishtirish uchun.

### Nega bu hisoblandi
Precision va Recall odatda **bir-biriga qarama-qarshi** ishlaydi: thresholdni pasaytirsangiz, ko'proq odamni "churn" deysiz → recall oshadi (kam odam sezilmay qoladi), lekin precision tushadi (ko'proq bekorga signal). Bularni alohida ko'rish, qaysi biri biznes uchun muhimroq ekanini tushunish uchun kerak (masalan churn masalasida odatda **recall muhimroq**, chunki mijozni yo'qotish xatosi qimmatroq).

### Qanday qarorga keldik
F1=0.594 — model o'rtacha darajada ishlayotganini ko'rsatdi, na juda yaxshi, na juda yomon. Bu raqam modelni **boshqa modellar bilan solishtirish** uchun bitta ko'rsatkich sifatida ishlatiladi.

### Agar bu qilinmaganda
Faqat accuracy va confusion matrix'ga qarab, thresholdni qanday sozlash kerakligini son bilan asoslab bo'lmas edi — Precision/Recall shu tanlovni miqdoriy qiladi.

---

## 4. ROC Curve (TPR va FPR)

### Nima qilindi
Har bir threshold (0 dan 1 gacha, 101 ta qadam) uchun:
```python
tpr = tp / (tp + fn)   # = Recall
fpr = fp / (fp + tn)
```
hisoblanib, grafik chizildi.

### Nega bu qilindi
Precision/Recall bitta thresholdga (masalan t=0.5) bog'liq. Lekin **qaysi threshold eng yaxshi** ekanini bilish uchun **barcha thresholdlarni birdaniga** ko'rish kerak. ROC curve aynan shuni qiladi — modelni bitta threshold tanlamasdan turib baholaydi.

### Bu nima degani
- **TPR (Y o'qi)** — chindan churn qiluvchilardan qanchasi to'g'ri topilyapti (Recall bilan bir xil).
- **FPR (X o'qi)** — chindan qolib ketuvchilardan qanchasi xato "churn" deb belgilanyapti.

Egri chiziq yuqoriga va chapga qanchalik "cho'zilgan" bo'lsa (ya'ni past FPR'da ham yuqori TPR bersa), model shuncha yaxshi.

### Qanday qarorga keldik
Model diagonal (tasodifiy taxmin) chizig'idan sezilarli yuqorida ekanligi ko'rildi — demak model haqiqatan **nimadir o'rganib olgan**, faqat tasodifga ishonib bashorat qilmayapti.

### Agar bu qilinmaganda
Faqat bitta threshold (masalan 0.5)dagi natijaga qarab model sifatini baholasak, aslida threshold=0.5 modelning eng yaxshi ishlashi bo'lmagan holat bo'lishi ham mumkin edi — ROC curve bu xavfni yo'q qiladi, chunki barcha variantlarni bir vaqtda ko'rsatadi.

---

## 5. Random va Ideal Model bilan solishtirish

### Nima qilindi
- **Random model** — `np.random.uniform(0, 1, ...)` — hech narsa o'rganmagan, tasodifiy ehtimollik beruvchi model.
- **Ideal model** — barcha manfiylarni (0) oldin, musbatlarni (1) keyin tartib bilan joylashtirilgan, xatosiz ajratuvchi "mukammal" model.

Uchalasi (haqiqiy model, random, ideal) bitta grafikda, FPR/TPR bo'yicha solishtirildi.

### Nega bu qilindi
ROC curve'ning o'zi "yaxshi" yoki "yomon"ligini bila olmaysiz, agar unga solishtirish uchun **pastki chegara (random)** va **yuqori chegara (ideal)** bo'lmasa. Bu ikkovi — **etalon** vazifasini o'taydi.

### Qanday qarorga keldik
Haqiqiy model:
- Random chizig'idan **ancha yuqorida** — demak model tasodifdan ancha yaxshi ishlayapti.
- Ideal chizig'idan **pastroq** — demak hali yaxshilanish imkoni bor (bu normal, hech qachon ideal modelga yetib bo'lmaydi).

Bu solishtirish **ROC-AUC** tushunchasini vizual tushunish uchun asos bo'ldi: AUC — bu ROC curve ostidagi maydon, 0.5 (random) dan 1.0 (ideal) gacha.

### Agar bu qilinmaganda
ROC curve'ni ko'rib "bu yaxshimi yomonmi" degan savolga aniq javob berib bo'lmas edi — nisbiy solishtirish yo'q bo'lganda raqamlar mavhum qolardi.

---

## 6. Cross-Validation (K-Fold)

### Nima qilindi
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    ...
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

# natija: 0.841 +- 0.006
```

`df_full_train` 5 ta teng qismga bo'lindi. Har safar 4 qismda model o'qitilib, 1 qismda tekshirildi — bu jarayon 5 marta takrorlandi (har safar boshqa qism validation bo'lib).

### Nega bu qilindi
Notebookning boshida bitta marta `train_test_split` qilingan edi (`random_state=42`). Lekin savol tug'iladi: **agar bo'linish boshqacha bo'lganda, natija ham boshqacha chiqarmidi?** Bitta bo'linishga tayanish — natijani **tasodifga** bog'liq qilib qo'yadi. Balki shu aynan bo'linishda model "omadli" yoki "omadsiz" chiqqandir.

### Qanday qarorga keldik
5 marta turli bo'linishda sinab ko'rilgach, natija **0.841 ± 0.006** chiqdi. Standart og'ish (0.006) juda kichik — bu degani, qaysi qismlar train/val bo'lishidan qat'iy nazar, model **barqaror** taxminan bir xil natija beradi. Demak, bizning boshlang'ich `random_state=42` bilan olingan natija ham ishonchli ekan (tasodifiy "omad" emas).

### Agar bu qilinmaganda
Bitta train/val bo'linishiga asoslanib xulosa chiqarilsa, va aynan o'sha bo'linish "qulay" (yoki "noqulay") chiqqan bo'lsa, modelning haqiqiy sifati haqida **noto'g'ri xulosa** qilingan bo'lardi — masalan aslida yomon model "yaxshi" ko'rinishi yoki aksincha.

---

## 7. C parametrini sozlash (Hyperparameter tuning)

### Nima qilindi
```python
for C in [0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
    # har biri uchun 5-fold cross-validation
```
Natija:
| C | AUC |
|---|---|
| 0.001 | 0.821 ± 0.006 |
| 0.01 | 0.838 ± 0.006 |
| 0.1 | 0.841 ± 0.006 |
| 0.5 – 10 | 0.841 ± 0.006 (deyarli o'zgarmaydi) |

### Bu nima
**C** — Logistic Regression'dagi **regularizatsiya kuchini** boshqaradi. 
- **Kichik C** (masalan 0.001) → kuchli regularizatsiya → model "oddiy"roq bo'lishga majburlanadi, ba'zan yetarlicha o'rganolmaydi (underfitting).
- **Katta C** → regularizatsiya kuchsizlanadi → model ma'lumotga ko'proq "erkin" moslashadi, lekin haddan tashqari katta bo'lsa overfitting xavfi bor.

### Nega bu qilindi
Model default `C=1.0` bilan o'qitilgan edi, lekin bu qiymat **eng yaxshisi ekani tekshirilmagan** edi. Turli C larni cross-validation orqali sinab, qaysi biri eng barqaror/yaxshi natija berishini bilish kerak edi.

### Qanday qarorga keldik
C=0.001 da natija sezilarli yomonlashadi (0.821) — demak juda kuchli regularizatsiya modelni zaiflashtiryapti. Lekin **C=0.1 dan boshlab** natija barqarorlashadi va deyarli o'zgarmaydi (0.841). Demak:
- Juda kichik C — zararli (underfitting).
- C=0.1 dan yuqori — farq yo'q, shuning uchun eng oddiy/ishonchli tanlov sifatida **C=1.0** qoldirildi (chunki qo'shimcha regularizatsiyadan foyda yo'q, lekin zarar ham yo'q).

### Agar bu qilinmaganda
Default C=1.0 "tasodifan" yaxshi chiqqan yoki yomon chiqqan bo'lishi mumkin edi, buni bilmay qolardik. Balki C=0.001 kabi yomon qiymat tanlangan bo'lsa, model kutilganidan ancha yomon ishlar edi, va buning sababini tushunmasdan qolardik.

---

## 8. Yakuniy model: full_train'da o'qitib, test'da baholash

### Nima qilindi
```python
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)   # = 0.862
```

### Nega bu qilindi
Cross-validation paytida har safar faqat `df_full_train`ning **4/5 qismida** o'qitilgan edi (5-fold bo'lgani uchun). Endi eng yaxshi parametr (C=1.0) tanlangach, modelni **butun `df_full_train`da** (ya'ni barcha mavjud train+val ma'lumotda, hech narsani "isrof qilmasdan") qayta o'qitish mumkin — ko'proq ma'lumot, odatda yaxshiroq model.

`df_test` esa shu vaqtgacha **umuman ishlatilmagan** edi — na o'qitishda, na parametr tanlashda. Shuning uchun bu — modelning **eng xolis, haqiqiy hayotdagi ishlashiga eng yaqin** bahosi.

### Qanday qarorga keldik
Test natijasi **AUC=0.862** — bu cross-validation'dagi o'rtacha natijadan (0.841) biroz yuqori chiqdi, chunki bu safar model ko'proq ma'lumotda (to'liq full_train) o'qitildi. Bu — modelning yakuniy, e'lon qilinadigan sifat ko'rsatkichi.

### Agar bu qilinmaganda (yoki test avvalroq ishlatilganda)
Agar test to'plami parametr tanlash (masalan C ni tanlashda) jarayonida ishlatilganda, model bilmasdan test'ga "moslashib qolar" edi — va test natijasi haqiqiy hayotdagi ishlashni emas, balki **soxta, haddan tashqari optimistik** natijani ko'rsatardi. Aynan shu sababdan test faqat **eng oxirida, bir marta** ishlatiladi — bu ML'dagi eng muhim qoidalardan biri.

---

## Umumiy xulosa: nega bu barcha qadamlar kerak edi

| Qadam | Qaysi savolga javob beradi |
|---|---|
| Accuracy + Dummy model | Model umuman nimadir o'rganganmi, yoki shunchaki ko'pchilik klassni takrorlayaptimi? |
| Confusion Matrix | Model qanday xato qilyapti — kimni yo'qotayapti, kimga bekorga signal beryapti? |
| Precision/Recall/F1 | Qaysi xato turi (FP yoki FN) biznes uchun muhimroq, va model shu bo'yicha qanday? |
| ROC Curve | Bitta thresholdga bog'lanmasdan, model umuman qanchalik yaxshi ajrata oladi? |
| Random/Ideal solishtirish | ROC natijasi "yaxshi" deganda — nisbatan nimaga nisbatan yaxshi? |
| Cross-Validation | Natija ishonchlimi, yoki tasodifiy bo'linishning natijasimi? |
| C parametrini sozlash | Model sozlamalari eng yaxshi holatga keltirilganmi? |
| Yakuniy test bahosi | Model haqiqiy hayotda (hech qachon ko'rmagan ma'lumotda) qanday ishlaydi? |

Har bir qadam — oldingi qadamdagi **cheklovni yopish** uchun qo'shilgan: accuracy yolg'on ishonch berdi → confusion matrix bilan tekshirildi → aniq metrikalar (precision/recall) bilan miqdorlashtirildi → ROC bilan umumlashtirildi → random/ideal bilan baholandi → cross-validation bilan ishonchliligi tasdiqlandi → C bilan optimallashtirildi → test bilan yakuniy, xolis baho olindi.
