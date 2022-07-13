# Deep Learning English Decoder Based On NLP (Natural Language Processing)
# 基於深度學習的英語解密自然語言處理模型
## Execution
```
in.txt : type encryption in in.txt.
out.txt : decryption will output to out.txt
Compile cypher_project.py and Use train.csv, test.csv, spam1.csv 
to train best-model.pt used in English validator.
the example encrypt text : UJLIALUGWLUWEMUMOVMNCNONCIHWCJBYL

```
## Frequency Validator
* 第⼀一部份為frequency validator，先得到⼀一般英⽂章中字⺟的出現頻率並排序 -> sort letter = ‘ETAOINSHRDLCUMWFGYPBVKJXQZ'
* 再來分析密⽂中的字⺟頻率(ansfre)並由⼤到小排序得到對照字典
```
-> normal English letters' frequency sorting = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
-> encryption's letters' frequency sorting   = 'CYUILPGKNDFHZQVAMBRSOXTJWE'
兩字串對照，每個字母生成一對一解密字典 ex:E->C, T->Y
```
* 接著iteration列出encrypt text's letters' frequency sorting字串的所有permutation
* 每個permutation用字典翻譯成可能的解密字串
## English validator
* 1.⽤re分割句⼦，再用flair做sentiment analysis得到stop word與情感分析，程式部分為去除亂碼與在a後空⼀格製造類似句⼦有空格的結構。同時引入flair做sentiment情感分析。在列表中增加兩個欄位
* 2.⾃己訓練nlp flair資料集，輸入字典來判斷句子是否為有效訊息，以此來找出正確的permutation解，⾃己製作spam1.csv訓練出best- model.pt，⽤這個model來判斷，spam代表是無效句子，⼿動輸入spam資料，兩種資料集分別標上label
* ![Variable Declaration](/img/c1.png)
* ![Variable Declaration](/img/c2.png)
* 3.再次用frequency analysis計算每筆密⽂可能解中⾼頻字⺟與低頻字母的出現比例
* 每筆資料設定新特徵值欄位freq
```
Sentence have e,t,a,o,i,n -> freq +1
Sentence have z,q,x,j,k,v -> freq -1
計算每筆密⽂可能解，將每筆資料之特徵值freq加入列表
```
## Language Detector 
* ⽤language detector模型檢測是否為英文
* 根據各國語言的特⾊音節、字⺟萃取特徵來分類語言，最後結果加入列表
```
Feature syllable of English:[a, i, o, s, t, an, d, e, er...]
```
* ![Variable Declaration](/img/c3.png)
## Choose Valid Sentence
* Choose English sentence 
* Check by English Validator (nlp model)
* Sort by freq
* ![Variable Declaration](/img/c4.png)
## Output Decryption to out.txt
* ![Variable Declaration](/img/c5.png)






