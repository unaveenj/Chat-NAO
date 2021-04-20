import sys
import argparse
import time
from naoqi import ALProxy

tts = audio = record = aup = None

def main(robotIP, port):
    global tts, audio, record, aup 
  # ----------> Connect to robot <----------
    tts = ALProxy("ALTextToSpeech", robotIP, port)
    audio = ALProxy("ALAudioDevice", robotIP, port)
    record = ALProxy("ALAudioRecorder", robotIP, port)
    aup = ALProxy("ALAudioPlayer", robotIP, port)
    #aup.stopAll()
  # ----------> recording <----------
##  print 'start recording...'
##  record_path = '/home/nao/record.wav'
##  record.startMicrophonesRecording(record_path, 'wav', 16000, (0,0,1,0))
##  time.sleep(10)
##  record.stopMicrophonesRecording()
##  print 'record over'


  # ----------> playing the recorded file <----------

    print 'play sound'
    music = '/home/nao/happy.mp3'
    fileID = aup.post.playFileFromPosition(music, 78, 1.0, 0)
    names = list()
    times = list()
    keys = list()
    

    try:
        ttsProxy = ALProxy("ALTextToSpeech",robotIP,port)
    except Exception,e:
        print("Could not create a proxy to ALTextToSpeech")

    #ttsProxy.say("...")

    names.append("HeadPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 4.95, 5.45, 5.95, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.172107, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.16418, [3, -0.15, 0], [3, 0.166667, 0]], [-0.172107, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.16418, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.172107, [3, -0.166667, 0.00639165], [3, 0.166667, -0.00639165]], [-0.20253, [3, -0.166667, 0.00913104], [3, 0.166667, -0.00913104]], [-0.226893, [3, -0.166667, 0], [3, 0.166667, 0]], [0.268781, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.432842, [3, -0.166667, 0.146303], [3, 0.166667, -0.146303]], [-0.60904, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.459022, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.49399, [3, -0.166667, 0], [3, 0.166667, 0]], [0.514872, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.584496, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.172107, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.204064, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.204064, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159578, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.174338, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159578, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.174338, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159578, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.174338, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159578, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.174338, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.172107, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("HeadYaw")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 4.95, 5.45, 5.95, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.00115764, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.0123138, [3, -0.15, 0], [3, 0.166667, 0]], [0.00115764, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0123138, [3, -0.166667, 0], [3, 0.166667, 0]], [0.00115764, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0107799, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.0014837, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.0014837, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.019984, [3, -0.166667, 0.0166242], [3, 0.166667, -0.0166242]], [-0.101229, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0798099, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.101229, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.069072, [3, -0.166667, -0.0170645], [3, 0.166667, 0.0170645]], [0.00115764, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.00832583, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.00832583, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.00832583, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.021518, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.00832583, [3, -0.166667, -0.00377927], [3, 0.166667, 0.00377927]], [0.00115764, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LAnklePitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.0859814, [3, -0.0166667, 0], [3, 0.15, 0]], [0.0873961, [3, -0.15, 0], [3, 0.166667, 0]], [0.0859814, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0873961, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0859814, [3, -0.166667, 0.0010227], [3, 0.166667, -0.0010227]], [0.08126, [3, -0.166667, 0], [3, 0.333333, 0]], [0.0812657, [3, -0.333333, 0], [3, 0.166667, 0]], [0.0812657, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0858622, [3, -0.166667, -0.000681164], [3, 0.333333, 0.00136233]], [0.0873961, [3, -0.333333, -0.000511328], [3, 0.333333, 0.000511328]], [0.0889301, [3, -0.333333, 0], [3, 0.166667, 0]], [0.0859814, [3, -0.166667, 0.00153403], [3, 0.166667, -0.00153403]], [0.0797259, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0797259, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0729001, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0729001, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0729001, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0729001, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0859814, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LAnkleRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.106353, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.099668, [3, -0.15, 0], [3, 0.166667, 0]], [-0.106353, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0996681, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.106353, [3, -0.166667, 0.00204535], [3, 0.166667, -0.00204535]], [-0.11194, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.105702, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.105702, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.108872, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.10427, [3, -0.333333, 0], [3, 0.333333, 0]], [-0.10427, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.106353, [3, -0.166667, 0.00153403], [3, 0.166667, -0.00153403]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861442, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861442, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861442, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.113474, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861442, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.106353, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LElbowRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.414982, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.392662, [3, -0.15, 0], [3, 0.166667, 0]], [-0.414982, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.392662, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.414982, [3, -0.166667, 0.0223193], [3, 0.166667, -0.0223193]], [-1.48794, [3, -0.166667, 0.0498419], [3, 0.166667, -0.0498419]], [-1.53778, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0433995, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0433995, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0214341, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.139552, [3, -0.333333, 0.0539457], [3, 0.333333, -0.0539457]], [-0.345108, [3, -0.333333, 0.0612066], [3, 0.166667, -0.0306033]], [-0.414982, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.401866, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.401866, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.391128, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.592401, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.391128, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.592401, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.391128, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.592401, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.391128, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.592401, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.414982, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LElbowYaw")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-1.19856, [3, -0.0166667, 0], [3, 0.15, 0]], [-1.17815, [3, -0.15, 0], [3, 0.166667, 0]], [-1.19856, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.17815, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.19856, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.339056, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.767852, [3, -0.166667, 0.00467571], [3, 0.166667, -0.00467571]], [-0.772528, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.772528, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.786984, [3, -0.166667, 0.0144559], [3, 0.333333, -0.0289118]], [-0.98487, [3, -0.333333, 0.0577807], [3, 0.333333, -0.0577807]], [-1.13367, [3, -0.333333, 0.0474858], [3, 0.166667, -0.0237429]], [-1.19856, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.20531, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.20531, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.20531, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.18276, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.20531, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.19856, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LHand")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.291965, [3, -0.0166667, 0], [3, 0.15, 0]], [0.3012, [3, -0.15, 0], [3, 0.166667, 0]], [0.291965, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3012, [3, -0.166667, 0], [3, 0.166667, 0]], [0.291965, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3076, [3, -0.166667, 0], [3, 0.166667, 0]], [0.29661, [3, -0.166667, 0], [3, 0.166667, 0]], [0.73, [3, -0.166667, 0], [3, 0.166667, 0]], [0.73, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3156, [3, -0.166667, 0.00299999], [3, 0.333333, -0.00599998]], [0.3096, [3, -0.333333, 0], [3, 0.333333, 0]], [0.3096, [3, -0.333333, 0], [3, 0.166667, 0]], [0.291965, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.352844, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.352844, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.352844, [3, -0.166667, 0], [3, 0.166667, 0]], [0.2952, [3, -0.166667, 0], [3, 0.166667, 0]], [0.352844, [3, -0.166667, 0], [3, 0.166667, 0]], [0.291965, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LHipPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.134022, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.248466, [3, -0.15, 0], [3, 0.166667, 0]], [0.134022, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.248467, [3, -0.166667, 0], [3, 0.166667, 0]], [0.134022, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0837758, [3, -0.166667, 0.0258277], [3, 0.166667, -0.0258277]], [-0.020944, [3, -0.166667, 0.0523599], [3, 0.166667, -0.0523599]], [-0.230383, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.230383, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.21165, [3, -0.166667, -0.0187333], [3, 0.333333, 0.0374666]], [-0.0475121, [3, -0.333333, -0.052156], [3, 0.333333, 0.052156]], [0.101286, [3, -0.333333, -0.040341], [3, 0.166667, 0.0201705]], [0.134022, [3, -0.166667, -0.00101159], [3, 0.166667, 0.00101159]], [0.135034, [3, -0.166667, 0], [3, 0.166667, 0]], [0.135034, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.331302, [3, -0.166667, 0], [3, 0.166667, 0]], [0.180661, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.331302, [3, -0.166667, 0], [3, 0.166667, 0]], [0.180661, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.331302, [3, -0.166667, 0], [3, 0.166667, 0]], [0.180661, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.331302, [3, -0.166667, 0], [3, 0.166667, 0]], [0.180661, [3, -0.166667, 0], [3, 0.166667, 0]], [0.134022, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LHipRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 4.95, 5.45, 5.95, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.113808, [3, -0.0166667, 0], [3, 0.15, 0]], [0.0138481, [3, -0.15, 0], [3, 0.166667, 0]], [0.113808, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0138481, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113808, [3, -0.166667, 0], [3, 0.166667, 0]], [0.107422, [3, -0.166667, 0.00558721], [3, 0.166667, -0.00558721]], [0.0802851, [3, -0.166667, 0.00830434], [3, 0.166667, -0.00830434]], [0.0575959, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0575959, [3, -0.166667, 0], [3, 0.166667, 0]], [0.124296, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0907571, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0997519, [3, -0.166667, -0.0206758], [3, 0.166667, 0.0206758]], [0.120428, [3, -0.166667, -0.00715868], [3, 0.166667, 0.00715868]], [0.142704, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113808, [3, -0.166667, 0.00331857], [3, 0.166667, -0.00331857]], [0.11049, [3, -0.166667, 0], [3, 0.166667, 0]], [0.11049, [3, -0.166667, 0], [3, 0.166667, 0]], [0.136568, [3, -0.166667, 0], [3, 0.166667, 0]], [0.08554, [3, -0.166667, 0], [3, 0.166667, 0]], [0.136568, [3, -0.166667, 0], [3, 0.166667, 0]], [0.08554, [3, -0.166667, 0], [3, 0.166667, 0]], [0.136568, [3, -0.166667, 0], [3, 0.166667, 0]], [0.08554, [3, -0.166667, 0], [3, 0.166667, 0]], [0.136568, [3, -0.166667, 0], [3, 0.166667, 0]], [0.08554, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113808, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LHipYawPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.165959, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.162562, [3, -0.15, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.162562, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0.00339676], [3, 0.166667, -0.00339676]], [-0.202446, [3, -0.166667, 0.0104466], [3, 0.166667, -0.0104466]], [-0.228638, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165806, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165806, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.15796, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.15796, [3, -0.333333, 0], [3, 0.333333, 0]], [-0.168698, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, -0.000767007], [3, 0.166667, 0.000767007]], [-0.164096, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.164096, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LKneePitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.0862456, [3, -0.0166667, 0], [3, 0.15, 0]], [0.133416, [3, -0.15, 0], [3, 0.166667, 0]], [-0.0862456, [3, -0.166667, 0], [3, 0.166667, 0]], [0.133416, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0862456, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.047596, [3, -0.166667, -0.0250038], [3, 0.333333, 0.0500077]], [0.138789, [3, -0.333333, 0], [3, 0.166667, 0]], [0.138789, [3, -0.166667, 0], [3, 0.166667, 0]], [0.128814, [3, -0.166667, 0.00997495], [3, 0.333333, -0.0199499]], [0.0337059, [3, -0.333333, 0.0327253], [3, 0.333333, -0.0327253]], [-0.067538, [3, -0.333333, 0.0266559], [3, 0.166667, -0.013328]], [-0.0862456, [3, -0.166667, 0.00276844], [3, 0.166667, -0.00276844]], [-0.0890141, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0890141, [3, -0.166667, 0], [3, 0.166667, 0]], [0.243864, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861126, [3, -0.166667, 0], [3, 0.166667, 0]], [0.243864, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861126, [3, -0.166667, 0], [3, 0.166667, 0]], [0.243864, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861126, [3, -0.166667, 0], [3, 0.166667, 0]], [0.243864, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861126, [3, -0.166667, 0.000133], [3, 0.166667, -0.000133]], [-0.0862456, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LShoulderPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[1.44407, [3, -0.0166667, 0], [3, 0.15, 0]], [1.15506, [3, -0.15, 0], [3, 0.166667, 0]], [1.44407, [3, -0.166667, 0], [3, 0.166667, 0]], [1.15506, [3, -0.166667, 0], [3, 0.166667, 0]], [1.44407, [3, -0.166667, 0], [3, 0.166667, 0]], [0.239262, [3, -0.166667, 0.120491], [3, 0.166667, -0.120491]], [0.118771, [3, -0.166667, 0.120491], [3, 0.166667, -0.120491]], [-1.60396, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.60396, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.24565, [3, -0.166667, -0.195257], [3, 0.333333, 0.390515]], [0.153358, [3, -0.333333, -0.415203], [3, 0.333333, 0.415203]], [1.24557, [3, -0.333333, -0.286826], [3, 0.166667, 0.143413]], [1.44407, [3, -0.166667, 0], [3, 0.166667, 0]], [1.43118, [3, -0.166667, 0], [3, 0.166667, 0]], [1.43118, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41891, [3, -0.166667, 0], [3, 0.166667, 0]], [1.49761, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41891, [3, -0.166667, 0], [3, 0.166667, 0]], [1.49761, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41891, [3, -0.166667, 0], [3, 0.166667, 0]], [1.49761, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41891, [3, -0.166667, 0], [3, 0.166667, 0]], [1.49761, [3, -0.166667, 0], [3, 0.166667, 0]], [1.44407, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LShoulderRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.230755, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.191792, [3, -0.15, 0], [3, 0.166667, 0]], [0.230755, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.191792, [3, -0.166667, 0], [3, 0.166667, 0]], [0.230755, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.243948, [3, -0.166667, 0], [3, 0.166667, 0]], [0.143502, [3, -0.166667, -0.00890659], [3, 0.166667, 0.00890659]], [0.152409, [3, -0.166667, 0], [3, 0.166667, 0]], [0.152409, [3, -0.166667, 0], [3, 0.166667, 0]], [0.190174, [3, -0.166667, -0.00383496], [3, 0.333333, 0.00766992]], [0.197844, [3, -0.333333, -0.00153399], [3, 0.333333, 0.00153399]], [0.199378, [3, -0.333333, -0.001534], [3, 0.166667, 0.000766999]], [0.230755, [3, -0.166667, -0.00894825], [3, 0.166667, 0.00894825]], [0.253067, [3, -0.166667, 0], [3, 0.166667, 0]], [0.253067, [3, -0.166667, 0], [3, 0.166667, 0]], [0.263807, [3, -0.166667, 0], [3, 0.166667, 0]], [0.199793, [3, -0.166667, 0], [3, 0.166667, 0]], [0.263807, [3, -0.166667, 0], [3, 0.166667, 0]], [0.199793, [3, -0.166667, 0], [3, 0.166667, 0]], [0.263807, [3, -0.166667, 0], [3, 0.166667, 0]], [0.199793, [3, -0.166667, 0], [3, 0.166667, 0]], [0.263807, [3, -0.166667, 0], [3, 0.166667, 0]], [0.199793, [3, -0.166667, 0], [3, 0.166667, 0]], [0.230755, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("LWristYaw")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.0956821, [3, -0.0166667, 0], [3, 0.15, 0]], [0.0843279, [3, -0.15, 0], [3, 0.166667, 0]], [0.0956821, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0843279, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0956821, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.625914, [3, -0.166667, 0.249683], [3, 0.166667, -0.249683]], [-1.40242, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.520057, [3, -0.166667, -0.371326], [3, 0.166667, 0.371326]], [0.825541, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.515466, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.30224, [3, -0.333333, -0.0787453], [3, 0.333333, 0.0787453]], [-0.042994, [3, -0.333333, -0.0884271], [3, 0.166667, 0.0442136]], [0.0956821, [3, -0.166667, -0.019326], [3, 0.166667, 0.019326]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.106489, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.106489, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.106489, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115008, [3, -0.166667, 0], [3, 0.166667, 0]], [0.106489, [3, -0.166667, 0.003221], [3, 0.166667, -0.003221]], [0.0956821, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RAnklePitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.0789405, [3, -0.0166667, 0], [3, 0.15, 0]], [0.0859461, [3, -0.15, 0], [3, 0.166667, 0]], [0.0789405, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0859461, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0789405, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0828779, [3, -0.166667, 0], [3, 0.333333, 0]], [0.0795622, [3, -0.333333, 0], [3, 0.166667, 0]], [0.0795622, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0798099, [3, -0.166667, 0], [3, 0.333333, 0]], [0.0782759, [3, -0.333333, 0.00153398], [3, 0.333333, -0.00153398]], [0.067538, [3, -0.333333, 0], [3, 0.166667, 0]], [0.0789405, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0767419, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0767419, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0844118, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0791605, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0844118, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0791605, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0844118, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0791605, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0844118, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0791605, [3, -0.166667, 0.000219911], [3, 0.166667, -0.000219911]], [0.0789405, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RAnkleRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.108629, [3, -0.0166667, 0], [3, 0.15, 0]], [0.108956, [3, -0.15, 0], [3, 0.166667, 0]], [0.108629, [3, -0.166667, 0], [3, 0.166667, 0]], [0.108956, [3, -0.166667, 0], [3, 0.166667, 0]], [0.108629, [3, -0.166667, 0.000326728], [3, 0.166667, -0.000326728]], [0.105888, [3, -0.166667, 0.000512778], [3, 0.333333, -0.00102556]], [0.104014, [3, -0.333333, 0], [3, 0.166667, 0]], [0.104014, [3, -0.166667, 0], [3, 0.166667, 0]], [0.101286, [3, -0.166667, 0], [3, 0.333333, 0]], [0.101286, [3, -0.333333, 0], [3, 0.333333, 0]], [0.101286, [3, -0.333333, 0], [3, 0.166667, 0]], [0.108629, [3, -0.166667, -0.00230102], [3, 0.166667, 0.00230102]], [0.115092, [3, -0.166667, 0], [3, 0.166667, 0]], [0.115092, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113558, [3, -0.166667, 0.00153396], [3, 0.166667, -0.00153396]], [0.103368, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113558, [3, -0.166667, 0], [3, 0.166667, 0]], [0.103368, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113558, [3, -0.166667, 0], [3, 0.166667, 0]], [0.103368, [3, -0.166667, 0], [3, 0.166667, 0]], [0.113558, [3, -0.166667, 0], [3, 0.166667, 0]], [0.103368, [3, -0.166667, 0], [3, 0.166667, 0]], [0.108629, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RElbowRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.410938, [3, -0.0166667, 0], [3, 0.15, 0]], [0.383542, [3, -0.15, 0], [3, 0.166667, 0]], [0.410938, [3, -0.166667, 0], [3, 0.166667, 0]], [0.383541, [3, -0.166667, 0], [3, 0.166667, 0]], [0.410938, [3, -0.166667, 0], [3, 0.166667, 0]], [0.403484, [3, -0.166667, 0], [3, 0.166667, 0]], [0.410861, [3, -0.166667, -0.00168883], [3, 0.166667, 0.00168883]], [0.413617, [3, -0.166667, 0], [3, 0.166667, 0]], [0.413617, [3, -0.166667, 0], [3, 0.166667, 0]], [0.392746, [3, -0.166667, 0], [3, 0.333333, 0]], [0.392746, [3, -0.333333, 0], [3, 0.333333, 0]], [0.392746, [3, -0.333333, 0], [3, 0.166667, 0]], [0.410938, [3, -0.166667, -0.0181918], [3, 0.166667, 0.0181918]], [1.32695, [3, -0.166667, -0.184627], [3, 0.166667, 0.184627]], [1.5187, [3, -0.166667, 0], [3, 0.166667, 0]], [0.452573, [3, -0.166667, 0.0755815], [3, 0.166667, -0.0755815]], [0.376991, [3, -0.166667, 0], [3, 0.166667, 0]], [0.452573, [3, -0.166667, 0], [3, 0.166667, 0]], [0.376991, [3, -0.166667, 0], [3, 0.166667, 0]], [0.452573, [3, -0.166667, 0], [3, 0.166667, 0]], [0.376991, [3, -0.166667, 0], [3, 0.166667, 0]], [0.452573, [3, -0.166667, 0], [3, 0.166667, 0]], [0.376991, [3, -0.166667, 0], [3, 0.166667, 0]], [0.410938, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RElbowYaw")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[1.20855, [3, -0.0166667, 0], [3, 0.15, 0]], [1.21028, [3, -0.15, 0], [3, 0.166667, 0]], [1.20855, [3, -0.166667, 0], [3, 0.166667, 0]], [1.21028, [3, -0.166667, 0], [3, 0.166667, 0]], [1.20855, [3, -0.166667, 0], [3, 0.166667, 0]], [1.21489, [3, -0.166667, 0], [3, 0.166667, 0]], [1.20455, [3, -0.166667, 0], [3, 0.166667, 0]], [1.21024, [3, -0.166667, 0], [3, 0.166667, 0]], [1.21024, [3, -0.166667, 0], [3, 0.166667, 0]], [1.19034, [3, -0.166667, 0], [3, 0.333333, 0]], [1.19034, [3, -0.333333, 0], [3, 0.333333, 0]], [1.19034, [3, -0.333333, 0], [3, 0.166667, 0]], [1.20855, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0413762, [3, -0.166667, 0], [3, 0.166667, 0]], [1.28085, [3, -0.166667, 0], [3, 0.166667, 0]], [0.64884, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41372, [3, -0.166667, 0], [3, 0.166667, 0]], [0.64884, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41372, [3, -0.166667, 0], [3, 0.166667, 0]], [0.64884, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41372, [3, -0.166667, 0], [3, 0.166667, 0]], [0.64884, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41372, [3, -0.166667, 0], [3, 0.166667, 0]], [1.20855, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RHand")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.302121, [3, -0.0166667, 0], [3, 0.15, 0]], [0.3088, [3, -0.15, 0], [3, 0.166667, 0]], [0.302121, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3088, [3, -0.166667, 0], [3, 0.166667, 0]], [0.302121, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3152, [3, -0.166667, 0], [3, 0.166667, 0]], [0.300403, [3, -0.166667, 0], [3, 0.166667, 0]], [0.300403, [3, -0.166667, 0], [3, 0.166667, 0]], [0.300403, [3, -0.166667, 0], [3, 0.166667, 0]], [0.3108, [3, -0.166667, 0], [3, 0.333333, 0]], [0.3108, [3, -0.333333, 0], [3, 0.333333, 0]], [0.3108, [3, -0.333333, 0], [3, 0.166667, 0]], [0.302121, [3, -0.166667, 0.00313334], [3, 0.166667, -0.00313334]], [0.292, [3, -0.166667, 0], [3, 0.166667, 0]], [0.292, [3, -0.166667, 0], [3, 0.166667, 0]], [0.9824, [3, -0.166667, 0], [3, 0.166667, 0]], [0.360237, [3, -0.166667, 0], [3, 0.166667, 0]], [0.9824, [3, -0.166667, 0], [3, 0.166667, 0]], [0.360237, [3, -0.166667, 0], [3, 0.166667, 0]], [0.9824, [3, -0.166667, 0], [3, 0.166667, 0]], [0.360237, [3, -0.166667, 0], [3, 0.166667, 0]], [0.9824, [3, -0.166667, 0], [3, 0.166667, 0]], [0.360237, [3, -0.166667, 0.058116], [3, 0.166667, -0.058116]], [0.302121, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RHipPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.124874, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.247016, [3, -0.15, 0], [3, 0.166667, 0]], [0.124874, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.247016, [3, -0.166667, 0], [3, 0.166667, 0]], [0.124874, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0837758, [3, -0.166667, 0.024303], [3, 0.166667, -0.024303]], [-0.020944, [3, -0.166667, 0.0523599], [3, 0.166667, -0.0523599]], [-0.230383, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.230383, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.207132, [3, -0.166667, -0.020821], [3, 0.333333, 0.0416421]], [-0.042994, [3, -0.333333, -0.0501107], [3, 0.333333, 0.0501107]], [0.0935321, [3, -0.333333, -0.0373041], [3, 0.166667, 0.018652]], [0.124874, [3, -0.166667, 0], [3, 0.166667, 0]], [0.110406, [3, -0.166667, 0], [3, 0.166667, 0]], [0.110406, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.329852, [3, -0.166667, 0], [3, 0.166667, 0]], [0.16006, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.329852, [3, -0.166667, 0], [3, 0.166667, 0]], [0.16006, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.329852, [3, -0.166667, 0], [3, 0.166667, 0]], [0.16006, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.329852, [3, -0.166667, 0], [3, 0.166667, 0]], [0.16006, [3, -0.166667, 0], [3, 0.166667, 0]], [0.124874, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RHipRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 5.95, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.118451, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.00609398, [3, -0.15, 0], [3, 0.166667, 0]], [-0.118451, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.00609397, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.118451, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.110406, [3, -0.166667, -0.00636094], [3, 0.166667, 0.00636094]], [-0.0802851, [3, -0.166667, -0.00880168], [3, 0.166667, 0.00880168]], [-0.0575959, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0575959, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.125746, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.10427, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.120428, [3, -0.166667, 0.00536903], [3, 0.166667, -0.00536903]], [-0.136484, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.118451, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.12728, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.12728, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.139552, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.117225, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.139552, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.117225, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.139552, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.117225, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.139552, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.117225, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.118451, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RHipYawPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.165959, [3, -0.0166667, 0], [3, 0.15, 0]], [-0.162562, [3, -0.15, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.162562, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0.00339676], [3, 0.166667, -0.00339676]], [-0.202446, [3, -0.166667, 0.0104466], [3, 0.166667, -0.0104466]], [-0.228638, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165806, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165806, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.15796, [3, -0.166667, 0], [3, 0.333333, 0]], [-0.15796, [3, -0.333333, 0], [3, 0.333333, 0]], [-0.168698, [3, -0.333333, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, -0.000767007], [3, 0.166667, 0.000767007]], [-0.164096, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.164096, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.256563, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.159943, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.165959, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RKneePitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.090787, [3, -0.0166667, 0], [3, 0.15, 0]], [0.131966, [3, -0.15, 0], [3, 0.166667, 0]], [-0.090787, [3, -0.166667, 0], [3, 0.166667, 0]], [0.131966, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.090787, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.055182, [3, -0.166667, -0.0255084], [3, 0.333333, 0.0510169]], [0.138789, [3, -0.333333, 0], [3, 0.166667, 0]], [0.138789, [3, -0.166667, 0], [3, 0.166667, 0]], [0.11816, [3, -0.166667, 0.0137119], [3, 0.333333, -0.0274238]], [0.0153821, [3, -0.333333, 0.032214], [3, 0.333333, -0.032214]], [-0.075124, [3, -0.333333, 0.0235931], [3, 0.166667, -0.0117966]], [-0.090787, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0889301, [3, -0.166667, 0], [3, 0.166667, 0]], [0.251617, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861124, [3, -0.166667, 0], [3, 0.166667, 0]], [0.251617, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861124, [3, -0.166667, 0], [3, 0.166667, 0]], [0.251617, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861124, [3, -0.166667, 0], [3, 0.166667, 0]], [0.251617, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.0861124, [3, -0.166667, 0.00467452], [3, 0.166667, -0.00467452]], [-0.090787, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RShoulderPitch")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[1.45155, [3, -0.0166667, 0], [3, 0.15, 0]], [1.16281, [3, -0.15, 0], [3, 0.166667, 0]], [1.45155, [3, -0.166667, 0], [3, 0.166667, 0]], [1.16281, [3, -0.166667, 0], [3, 0.166667, 0]], [1.45155, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41746, [3, -0.166667, 0], [3, 0.166667, 0]], [1.44184, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41416, [3, -0.166667, 0], [3, 0.166667, 0]], [1.41416, [3, -0.166667, 0], [3, 0.166667, 0]], [1.42359, [3, -0.166667, -0.00224095], [3, 0.333333, 0.00448189]], [1.43433, [3, -0.333333, -0.00357931], [3, 0.333333, 0.00357931]], [1.44507, [3, -0.333333, -0.00382551], [3, 0.166667, 0.00191276]], [1.45155, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.124212, [3, -0.166667, 0], [3, 0.166667, 0]], [0.467912, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.57603, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.0088, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.57603, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.0088, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.57603, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.0088, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.57603, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.0088, [3, -0.166667, -0.504597], [3, 0.166667, 0.504597]], [1.45155, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RShoulderRoll")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[-0.229928, [3, -0.0166667, 0], [3, 0.15, 0]], [0.191708, [3, -0.15, 0], [3, 0.166667, 0]], [-0.229928, [3, -0.166667, 0], [3, 0.166667, 0]], [0.191709, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.229928, [3, -0.166667, 0.0140201], [3, 0.166667, -0.0140201]], [-0.243948, [3, -0.166667, 0.0035116], [3, 0.166667, -0.0035116]], [-0.250998, [3, -0.166667, 0.00704955], [3, 0.166667, -0.00704955]], [-0.288126, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.288126, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.285366, [3, -0.166667, -0.00115889], [3, 0.333333, 0.00231779]], [-0.277696, [3, -0.333333, -0.00332367], [3, 0.333333, 0.00332367]], [-0.265424, [3, -0.333333, -0.0106151], [3, 0.166667, 0.00530755]], [-0.229928, [3, -0.166667, -0.0354961], [3, 0.166667, 0.0354961]], [0.240796, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0260359, [3, -0.166667, 0.0641723], [3, 0.166667, -0.0641723]], [-0.144238, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.110153, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.144238, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.110153, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.144238, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.110153, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.144238, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.110153, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.229928, [3, -0.166667, 0], [3, 0, 0]]])

    names.append("RWristYaw")
    times.append([0, 0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95, 4.45, 5.45, 6.45, 6.95, 7.45, 7.95, 8.45, 8.95, 9.45, 9.95, 10.45, 10.95, 11.45, 11.95, 12.45])
    keys.append([[0.104823, [3, -0.0166667, 0], [3, 0.15, 0]], [0.101202, [3, -0.15, 0], [3, 0.166667, 0]], [0.104823, [3, -0.166667, 0], [3, 0.166667, 0]], [0.101202, [3, -0.166667, 0], [3, 0.166667, 0]], [0.104823, [3, -0.166667, 0], [3, 0.166667, 0]], [0.0689881, [3, -0.166667, 0], [3, 0.166667, 0]], [0.105009, [3, -0.166667, 0], [3, 0.166667, 0]], [0.105009, [3, -0.166667, 0], [3, 0.166667, 0]], [0.105009, [3, -0.166667, 0], [3, 0.166667, 0]], [0.10427, [3, -0.166667, 0], [3, 0.333333, 0]], [0.10427, [3, -0.333333, 0], [3, 0.333333, 0]], [0.10427, [3, -0.333333, 0], [3, 0.166667, 0]], [0.104823, [3, -0.166667, -0.000553097], [3, 0.166667, 0.000553097]], [0.340507, [3, -0.166667, -0.14436], [3, 0.166667, 0.14436]], [0.970981, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.549213, [3, -0.166667, 0.377669], [3, 0.166667, -0.377669]], [-1.29503, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.549213, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.29503, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.549213, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.29503, [3, -0.166667, 0], [3, 0.166667, 0]], [-0.549213, [3, -0.166667, 0], [3, 0.166667, 0]], [-1.29503, [3, -0.166667, 0], [3, 0.166667, 0]], [0.104823, [3, -0.166667, 0], [3, 0, 0]]])


    try:
      motion = ALProxy("ALMotion",robotIP,port)
      motion.angleInterpolationBezier(names, times, keys)
    except BaseException, err:
      print err

    aup.stopAll()
    
if __name__ == "__main__":

    robotIP = "192.168.210.156"#"192.168.11.3"

    port = 9559 #9559 # Insert NAO port


    if len(sys.argv) <= 1:
        print "(robotIP default: 127.0.0.1)"
    elif len(sys.argv) <= 2:
        robotIP = sys.argv[1]
    else:
        port = int(sys.argv[2])
        robotIP = sys.argv[1]

    main(robotIP, port)
