# Neural Disentanglement for Text Attributes

## Datasets

* Multi-Domain review set
    * category: home-and-kitchen; stats: {1: 418381, 2: 242048, 3: 345094, 4: 740864, 5: 2507539}
    * category: electronics; stats: {1: 108725, 2: 82139, 3: 142257, 4: 347041, 5: 1009026}

## Evaluations Done

* Test without BoW loss (Amazon Reviews Multi)
    * Without BoW: S=0.68, C=0.92
    * With BoW: S=0.63, C=0.92
* Filter stop-words and sentiment words (Amazon Reviews Multi)
    * Stop words provided by NLTK, sentiment words from SentiWordnet
    * style-transfer - 0.72
    * content-preservation - 0.89
* Test on Pang & Lee set (Movie Reviews)
    * style-transfer: 0.727173
    * content-preservation: 0.91
* Style Embedding entire vector dropout (Amazon Reviews Multi)
    * Without: (20180424190136)
        * style-transfer - 0.81
        * content-preservation - 0.93
    * With: (20180424234705),
        * style-transfer - 0.57
        * content-preservation - 0.97
* New evaluations without BoW:
    * Dataset Details:
        * Train/Validation/Test - 131072/2048/32768
    * 20180429190944
        * 300 enc/dec, 32 style, 256 content
        * style-transfer - 0.49
        * content-preservation - 0.91
    * 20180430200541
        * 1024 enc/dec, 64 style, 512 content
        * style-transfer - 0.56
        * content-preservation - 0.92
    * 20180501182818
        * 1024 enc/dec, 64 style, 512 content, increase recurrent dropout to 0.6, 25 epochs
        * style-transfer - 0.58
        * content-preservation - 0.90
    * 20180502094334
        * 1024 enc/dec, 64 style, 512 content, increase recurrent dropout to 0.6, 50 epochs, remove gaussian noise
        * style-transfer - 0.60
        * content-preservation - 0.91
    * 20180503234630
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **10 epochs**, no gaussian noise, $10^{-4}$ learning rate, no pretrained embeddings
        * style-transfer - 0.80
        * content-preservation - 0.90
    * 20180504011512
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 20 epochs, no gaussian noise, $10^{-4}$ learning rate, no pretrained embeddings
        * style-transfer - 0.82
        * content-preservation - 0.90
    * 20180504150849
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 25 epochs, no gaussian noise, $5 * 10^{-3}$ learning rate for discriminator, $10^{-4}$ learning rate for the overall model, no pretrained embeddings
        * style-transfer - 0.76
        * content-preservation - 0.90
    * 20180504192623
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 25 epochs, no gaussian noise, $3 * 10^{-3}$ **learning rate** for discriminator, $3 * 10^{-3}$ learning rate for the overall model, no pretrained embeddings
        * style-transfer - 0.73
        * content-preservation - 0.91
    * 20180504233917
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 25 epochs, no gaussian noise, $3 * 10^{-3}$ learning rate for discriminator, $3 * 10^{-3}$ learning rate for the overall model, **with pretrained embeddings**
        * style-transfer - 0.73
        * content-preservation - 0.91
* Saved Models
    * 20180505195732
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 10 epochs, no gaussian noise, $3 * 10^{-3}$ learning rate for discriminator, $3 * 10^{-3}$ learning rate for the overall model, **with pretrained embeddings**
        * style-transfer - 0.8
        * content-preservation - 0.9
    * 20180505214502
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, 10 epochs, no gaussian noise, $3 * 10^{-3}$ learning rate for discriminator, $3 * 10^{-3}$ learning rate for the overall model, with **pretrained embeddings (amazon)**
        * style-transfer - 0.7966613769531
        * content-preservation - 0.9000228201200436
    * 20180506001538
        * 256 enc/dec, **128 style**, 128 content, all dropout 0.8, 10 epochs, no gaussian noise, $3 * 10^{-3}$ learning rate for discriminator, $3 * 10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon)
        * style-transfer - 0.8224182128906
        * content-preservation - 0.8930211180683600
    * 20180506235458
        * 256 enc/dec, **32 style**, 128 content, all dropout 0.8, 10 epochs, no gaussian noise, $3 * 10^{-3}$ learning rate for discriminator, $3 * 10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon)
        * style-transfer - 0.789886474609
        * content-preservation - 0.898045495768454
    * 20180507210445 
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **50 epochs**, no gaussian noise, $10^{-3}$ learning rate for discriminator, $10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon)
        * style-transfer - 0.816818
        * content-preservation - 0.9079977952003060
    * 20180508200351
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **100 epochs**, no gaussian noise, $10^{-3}$ learning rate for discriminator, $10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon)
        * style-transfer - 0.71
        * content-preservation - 0.91
* Ablation tests:
    * 20180509154733
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **50 epochs**, no gaussian noise, $10^{-3}$ learning rate for discriminator, $10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon), **only reconstruction**
        * style-transfer - 0.505340576171875
        * content-preservation - 0.9103426711684577
    * 20180510131508
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **50 epochs**, no gaussian noise, $10^{-3}$ learning rate for discriminator, $10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon), **reconstruction + adversarial**
        * style-transfer - 0.590148925781250
        * content-preservation - 0.9121321414316516
    * 20180510164629
        * 256 enc/dec, 8 style, 128 content, all dropout 0.8, **50 epochs**, no gaussian noise, $10^{-3}$ learning rate for discriminator, $10^{-3}$ learning rate for the overall model, with pretrained embeddings (amazon), **reconstruction + multi-task**
        * style-transfer - 0.64451599121094
        * content-preservation - 0.9053788464373388
* AAAI Paper code results
    * 64 style embedding, 50 epochs
    * AAAI model
        * transfer-strength: 0.400909423828
        * content-preservation:  0.92467149687875
        * word-overlap: 0.15145071786919460
    * Ours (20180513173318)
        * transfer-strength: 0.797790527343750
        * content-preservation: 0.8904156592216974
        * word-overlap: 0.022897249702992978
* With nearest neighbour algorithm, 0.001 lr, 8 style embedding, 10 epochs
    * transfer-strength: 0.695068
    * content-preservation: 0.892879759797933
    * word-overlap: 0.05431860391012187
* With nearest neighbour algorithm, 0.001 lr, 64 style embedding, 50 epochs
    * transfer-strength: 0.716309
    * content-preservation: 0.895156093744239
    * word-overlap: 0.030118052270010074
* 20180516181400 - Revert to mean style vector, 0.001 lr, 8 style embedding, 10 epochs
    * transfer-strength: 0.754913330078125
    * content-preservation: 0.8976162624090924
    * word-overlap: 0.06057908695838731
    * classifier-style: 0.78369140625
    * classifier-overall: 0.78204345703125
* 20180516215621 - mean style vector, 0.001 lr, 8 style embedding, 10 epochs, style dropout
    * transfer-strength: 0.7422943115234375
    * content-preservation: 0.8999875910922805
    * word-overlap: 0.05786270631990821
    * classifier-style: 0.78662109375
    * classifier-overall: 0.7857666015625
* 20180517010625 - mean style vector, 0.001 lr, 8 style embedding, 20 epochs, style dropout, beam search
    * transfer-strength: 0.745086669921875
    * content-preservation: 0.8953697162057292
    * word-overlap: 0.07181230309444453
    * classifier-style: 0.783355712890625
    * classifier-overall: 0.78326416015625
* NIPS Paper code results
    * 20 epochs
    * NIPS model
        * transfer-strength: 0.4708251953125
        * content-preservation: 0.8823620196709927
        * word-overlap: 0.007903890731720382
    * AAAI model
        * transfer-strength: 0.400909423828
        * content-preservation: 0.92467149687875
        * word-overlap: 0.15145071786919462
* Revert to higher dimensions for the style-transfer model - better to stick with lower dimensions
* Try removing gaussian noise from the discriminator inputs - seems to have almosts no effect
* Change T-SNE interface file to randomly choose a small amount of points as opposed to all of the points. - Much better to visualize
* Train Amazon specific word vector model - Doesn't seem to be a big improvement over the Word2Vec model
* Training with a learning rate of 0.003 collapses near the 25 epoch mark
* content-preservation metric testing
    * 0.8911610388259674 with no filtering
    * 0.8836801496804587 with opinion lexicon filtering
    * 0.7315150589751056 with opinion lexicon and stopwords filtering
* Added sentiment word exclusion before content-preservation
* Implement new content-preservation metric - word overlap
* Get AAAI paper code working
* Use the nearest neighbor of in the style space for style reconstruction
* Use content embedding vector size of 2 to check if T-SNE is obscuring separation - NOT NEEDED
* Run the sentiment classification task. Train a logistic regression on encoded space, same train/val/test split
* Greedy vs. Beam Search

## Final Experiments

### 10 Epoch Experiments
* (20180518201624-training) : k nearest neighbour
    * transfer-strength: 0.74658203125
    * content-preservation: 0.8875628237232267
    * word-overlap: 0.05394392189374524
    * classifier-style: 0.7998046875
    * classifier-adversarial: 0.58203125
    * classifier-overall: 0.80029296875
* (20180518213443-training) : mean style vector
    * transfer-strength: 0.7910614013671875
    * content-preservation: 0.8877433668021131
    * word-overlap: 0.05437092475610381
    * classifier-adversarial: 0.538848876953125
    * classifier-style: 0.789794921875
    * classifier-overall: 0.78997802734375
* (20180518222439-training) : mean style vector, 32 style
    * transfer-strength: 0.7946014404296875
    * content-preservation: 0.8797096846630614
    * word-overlap: 0.04883826285278362
    * classifier-adversarial: 0.55084228515625
    * classifier-style: 0.788543701171875
    * classifier-overall: 0.78826904296875
* (20180519022925-training) : mean style vector, 128 style
    * transfer-strength: 0.8749847412109375
    * content-preservation: 0.8711442245357672
    * word-overlap: 0.006109403522368188
    * classifier-style: 0.604736328125
    * classifier-adversarial: 0.78515625
    * classifier-overall: 0.78472900390625

### 50 Epoch Experiments
* (20180519114126-training, 20180519220554-inference) : k nearest neighbour
    * transfer-strength: 0.72216796875
    * content-preservation: 0.894967571426672
    * word-overlap: 0.0747200648391525
    * classifier-adversarial: 0.57470703125
    * classifier-style: 0.79443359375
    * classifier-overall: 0.7939453125
* (20180519114126-training, 20180519214142-inference) : mean style vector
    * transfer-strength: 0.7708587646484375
    * content-preservation: 0.8958720013700059
    * word-overlap: 0.07680581206912761
    * classifier-adversarial: 0.569091796875
    * classifier-style: 0.781768798828125
    * classifier-overall: 0.781524658203125
* (20180519151229-training, 20180520031503-inference) : mean style vector, 32 style
    * transfer-strength: 0.8507080078125
    * content-preservation: 0.8944877526990274
    * word-overlap: 0.04919787621545192
    * classifier-adversarial: 0.54644775390625
    * classifier-style: 0.774932861328125
    * classifier-overall: 0.774932861328125
* (20180519151339-training, 20180520032534-inference) : mean style vector, 128 style
    * transfer-strength: 0.880157470703125
    * content-preservation: 0.8781093531214081
    * word-overlap: 0.010374413246302285
    * classifier-adversarial: 0.567596435546875
    * classifier-style: 0.772918701171875
    * classifier-overall: 0.773284912109375

### Comparison Tests
* AAAI model
    * transfer-strength: 0.400909423828
    * content-preservation: 0.92467149687875, 0.9241252190839184
    * word-overlap: 0.15145071786919462
* NIPS model
    * transfer-strength: 0.4609375
    * content-preservation: 0.8830481742906375
    * word-overlap: 0.01140765251418234

## Last Minute tests

AAAI
/home/v2john/text_style_transfer/model/data
Ours - 20180522125734

* Ours
    * transfer-strength: 0.7653645833333333
    * content-preservation: 0.8870463371015553
* AAAI model
    * transfer-strength: 0.421
    * content-preservation: 0.9203019136115124
* NIPS model
    * transfer-strength: 0.45977372029803765
    * content-preservation: 0.8838310824209868

---

## Experiments on the complete AAAI Amazon review dataset

* AAAI
    * transfer-strength: 0.4605
    * content-preservation: 0.9302606227763774
* Ours (20180525233145-training, 20180526114517-inference)
    * transfer-strength: 0.7417263732514501
    * content-preservation: 0.9025863818128192
    * word-overlap: 0.0995391518533518
* Ours (20180526192804-training, 20180527154217-inference): Style embedding = 2
    * transfer-strength: 0.6138537286224413
    * content-preservation: 0.9056685760316792
    * word-overlap: 0.12152895890064763
* Ours (20180527030609-training, 20180527154323-inference), Style embedding = 2, Learning rate = 10^-4, batch size 64, lambda weights = 0.1
    * transfer-strength: 0.76011510088837
    * content-preservation: 0.8946310404082443
    * word-overlap: 0.07102844855565332
* Ours (20180529131309-training, output/20180530122711-inference), batch_size = 32
    * transfer-strength: 0.6694461370064764
    * content-preservation: 0.9029985146333501
    * word-overlap: 0.11182601382709641
* Ours (20180530182346-training, 20180531132104-inference), with BoW loss, lambda = 0.0001
    * transfer-strength: 0.7008537516346147
    * content-preservation: 0.9029972591947888
    * word-overlap: 0.11600832457044769

---

## Experiments on the Yelp Dataset

* Ours (saved-models/20180601160201, output/20180602002719-inference), with BoW loss, lambda = 0.0001
    * transfer-strength: 0.7134456560178669
    * content-preservation: 0.8881922801244286
    * word-overlap: 0.2644665557287217
* MIT Paper
    * transfer-strength: 0.8086700035456021
    * content-preservation: 0.8919784410198304
    * word-overlap: 0.20868662367654828

### Variational
* KL loss goes to zero and the decoder produces the same sentence always for weights = 1

* Ours(saved-models/20180602210347, output/20180603140501-inference) 
    * without BoW loss
    * style_kl_loss_weight = 0.001, content_kl_loss_weight = 0.0003 (no batch normalization)
    * transfer-strength: 0.9548472999287226
    * content-preservation: 0.7028374261746729
    * word-overlap: 0.03665597813113504
