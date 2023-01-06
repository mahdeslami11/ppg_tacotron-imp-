# PPG_Tacotron : Many-to-One Voice Conversion with Non-Parallel Data
## Deep-Voice-Conversion in pytorch
> Original Authors : Dabi Ahn

> Original Project URL : https://github.com/andabi/deep-voice-conversion

## Intro
Thanks to the original Dabi Ahn for his voice conversion project on GitHub: Deep-Voice-Conversion.
PPG_Tacotron project is an implementation of the Deep-Voice-Conversion project based on PyTorch.
This implementation based on PyTorch improved the training speed of the model to 9x, and the generated speech quality was consistent with the original project

## Model Architecture
This is a many-to-one voice conversion system,
 which adopts the speech conversion model proposed by ICME2016 : [Phonetic posteriorgrams for many-to-one voice conversion without parallel data training](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 
 CBHG module and PreNet module in [Tacotron](https://arxiv.org/abs/1703.10135) model were used to optimize the model
 
The main significance of this work is that we could generate a target speaker's utterances without parallel data like <source's wav, target's wav>, <wav, text> or <wav, phone>, but only waveforms of the target speaker.
(To make these parallel datasets needs a lot of effort.)
All we need in this project is a number of waveforms of the target speaker's utterances and only a small set of <wav, phone> pairs from a number of anonymous speakers.

The model architecture consists of two modules:
1. Net1(phoneme classification) classify someone's utterances to one of phoneme classes at every timestep.
    * Phonemes are speaker-independent while waveforms are speaker-dependent.
2. Net2(speech synthesis) synthesize speeches of the target speaker from the phones.

### Net1 is a classifier.
* Process: wav -> spectrogram -> mfccs -> phoneme dist.
* Net1 classifies spectrogram to phonemes that consists of 60 English phonemes at every timestep.
  * For each timestep, the input is log magnitude spectrogram and the target is phoneme dist.
* Objective function is cross entropy loss.
* [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1) used.
  * contains 630 speakers' utterances and corresponding phones that speaks similar sentences.
* Over 70% test accuracy

### Net2 is a synthesizer.
Net2 contains Net1 as a sub-network.
* Process: net1(wav -> spectrogram -> mfccs -> phoneme dist.) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input/target is a set of target speaker's utterances.
* Since Net1 is already trained in previous step, the remaining part only should be trained in this step.
* Loss is reconstruction error between input and target. (L2 distance)
* Datasets
    * Target(anonymous female): [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
* Griffin-Lim reconstruction when reverting wav from spectrogram.

## Implementations
### Requirements
* python 3.7
* pytorch == 1.5
* librosa == 0.7.2

### Settings
* sample rate: 16,000Hz
* window length: 25ms
* hop length: 5ms

### Procedure
* Train phase: Net1 and Net2 should be trained sequentially.
  * Train1(training Net1)
    * Run `train_net1.py` to train and `test_net1.py` to test.
  * Train2(training Net2)
    * Run `train_net2.py` to train and `test_net2.py` to test.
      * Train2 should be trained after Train1 is done!
* Convert phase: feed forward to Net2
    * Run `convert.py` to get result samples.
    * Check Tensorboard's audio tab to listen the samples.
    * Take a look at phoneme dist. visualization on Tensorboard's image tab.
      * x-axis represents phoneme classes and y-axis represents timesteps
      * the first class of x-axis means silence.

## Tips (Lessons We've learned from this project)
* Window length and hop length have to be small enough to be able to fit in only a phoneme.
* Obviously, sample rate, window length and hop length should be same in both Net1 and Net2.
* Before ISTFT(spectrogram to waveforms), emphasizing on the predicted spectrogram by applying power of 1.0~2.0 is helpful for removing noisy sound.
* It seems that to apply temperature to softmax in Net1 is not so meaningful.
* IMHO, the accuracy of Net1(phoneme classification) does not need to be so perfect.
  * Net2 can reach to near optimal when Net1 accuracy is correct to some extent.

## References
* ["Phonetic posteriorgrams for many-to-one voice conversion without parallel data training"](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 2016 IEEE International Conference on Multimedia and Expo (ICME)
* ["TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS"](https://arxiv.org/abs/1703.10135), Submitted to Interspeech 2017


This paper proposes a novel approach to voice conversion with non-parallel training data. The idea is to bridge between speakers by means of Phonetic PosteriorGram- s (PPGs) obtained from a speaker-independent automatic speech recognition (SI-ASR) system. It is assumed that these PPGs can represent articulation of speech sounds in a speaker- normalized space and correspond to spoken content speaker- independently. The proposed approach first obtains PPGs of target speech. 

![4DC635A0-E031-4930-BE5C-3F29AE6C9744](https://user-images.githubusercontent.com/115027808/211002774-daf49d68-34ce-4c0b-a8b0-095137419fad.jpeg)

Then, a Deep Bidirectional Long Short- Term Memory based Recurrent Neural Network (DBLSTM) structure is used to model the relationships between the PPGs and acoustic features of the target speech. To convert arbitrary source speech, we obtain its PPGs from the same SI-ASR and feed them into the trained DBLSTM for generating converted speech. Our approach has two main advantages: 1) no parallel training data is required; 2) a trained model can be applied to any other source speaker for a fixed target speaker (i.e., many- to-one conversion). Experiments show that our approach performs equally well or better than state-of-the-art systems in both speech quality and speaker similarity.
![D0E5A95B-5A2C-4100-9ADB-C14EAB81040D](https://user-images.githubusercontent.com/115027808/211002908-cc08c670-4e5a-4596-b013-1981c19d4650.jpeg)

main project: https://github.com/LJY-M/ppg_tacotron

I made these changes after studying these valuable articles:
* Kazuhiro Kobayashi, Tomoki Toda, Graham Neubig, Sakriani Sakti, and Satoshi Nakamura, “Statistical singing voice conver- sion based on direct waveform modification with global vari- ance,” Proceedings of the Annual Conference of the Interna- tional Speech Communication Association, INTERSPEECH, vol. 2015-January, no. September, pp. 2754–2758, 2015.
* Kazuhiro Kobayashi, Tomoki Toda, Graham Neubig, Sakriani Sakti, and Satoshi Nakamura, “Statistical Singing Voice Con- version with direct Waveform modification based on the Spec- trum Differential,” Interspeech 2014, 2014.
* Fernando Villavicencio and Jordi Bonada, “Applying Voice Conversion To Concatenative Singing-Voice Synthesis,” Inter- speech 2010, 2010.

Mehran Rajaeifar [iau.stb]
