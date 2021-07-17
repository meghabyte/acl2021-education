import numpy as np
import torch
from sklearn import metrics
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt

#These are the IDs of questions in the test set that are not included in the Duolingo training set (not see ever for any student)
french_unseen = [11, 13, 15, 24, 25, 26, 27, 28, 29, 31, 181, 443, 444, 447, 455, 456, 458, 461, 474, 830, 897, 898, 919, 920, 927, 968, 980, 988, 1280, 1324, 1333, 1344, 1545, 1965, 1969, 1977, 1981, 1993, 2030, 2145, 2258, 2271, 2404, 3071, 3106, 3110, 3120, 3121, 3122, 3136, 3139, 3144, 3467, 3788, 3790, 3792, 3841, 3844, 3861, 3863, 3876, 3877, 3878, 3883, 4100, 4213, 4214, 4225, 4758, 4882, 4906, 4920, 5268, 6131, 6134, 6644, 6898, 6998, 7003, 7005, 7007, 7009, 7011, 7013, 7014, 7015, 7020, 7025, 7028, 7029, 7031, 7036, 7043, 7064, 7070, 7071, 7072, 7398, 7439, 7440, 7448, 7610, 8200, 8208, 8212, 8214, 8215, 8216, 8217, 8218, 8219, 8220, 8222, 8229, 8234, 8235, 8237, 8241, 8242, 8243, 8245, 8258, 8267, 8268, 8269, 8272, 8273, 8277, 8280, 8290, 8291, 8292, 8295, 9516, 9794, 10093, 10097, 10140, 10156, 10167, 10176, 10341, 11329, 12062, 12090, 12094, 12104, 12107, 12108, 12118, 12122, 12143, 12431, 12437, 13035, 13167, 13171, 13175, 13181, 13182, 13187, 13208, 13255, 13335, 13338, 13340, 13344, 13347, 13362, 14422, 14433, 14434, 14441, 14837, 14846, 15345, 15443, 15521, 15534, 15541, 15619, 15624, 15645, 15722, 16182, 16212, 16221, 16228, 16241, 16491, 16680, 16681, 16682, 16683, 16685, 16686, 16687, 16698, 16713, 16714, 16732, 16735, 16736, 16737, 16740, 16742, 16749, 16750, 16751, 16764, 16765, 16766, 16768, 16769, 16770, 17040, 17042, 17356, 17363, 17705, 17717, 17723, 17730, 17732, 18275, 18533, 18536, 18537, 18552, 18555, 18579, 18715, 19099, 19125, 19176, 19999, 20015, 20512, 20626, 20627, 20630, 20633, 20634, 20642, 20645, 20650, 20651, 20653, 20655, 20657, 20658, 20659, 20660, 20662, 20663, 20665, 20666, 20667, 20669, 20671, 20674, 20676, 20678, 20680, 20682, 20709, 20928, 21350, 21595, 21601, 21827, 21848, 22229, 23374, 23877, 23901, 24070, 24696, 24705, 24717, 24730, 24732, 24733, 25043, 25052, 25073, 25079, 25129, 25772, 25783, 25845, 25932, 25952, 25988, 25994, 26004, 26005, 26010, 26011, 26015, 26016, 26017, 26020, 26025, 26034, 26442, 26472, 26731, 27169, 27172, 27174, 27205, 27215, 27230, 27688, 27695, 27697, 28051, 28063, 28081, 28086, 28105, 28113, 28122, 28327, 28340, 28346, 28348, 28349, 28378, 28398, 28711, 28718, 28746, 29005, 29220, 29387, 29539, 29774, 29793, 29912, 29917, 29923, 29924, 29932, 29946, 29950, 29981, 30008, 30009, 30397, 30502, 30515, 30519, 30547, 30548, 30647, 30650, 30692, 31280, 31591, 31618, 31680, 31729]
spanish_unseen = [89, 555, 880, 1302, 1325, 2381, 2579, 3220, 3588, 3591, 3884, 4324, 4325, 4326, 4361, 4370, 4390, 4391, 4406, 4442, 4451, 4778, 4786, 4798, 4816, 5762, 5928, 6928, 6930, 6933, 6949, 6950, 6952, 6953, 6954, 6957, 6958, 6959, 6962, 6970, 6971, 6977, 6985, 6986, 6987, 6991, 6995, 7109, 7114, 8232, 9107, 9313, 9757, 9832, 10096, 10162, 10691, 10946, 10957, 10984, 11007, 11029, 11401, 11403, 11424, 11550, 11563, 11573, 11576, 11595, 11596, 11628, 11681, 11693, 11694, 11695, 11699, 11735, 11850, 11862, 11863, 11866, 11877, 11927, 12181, 12297, 12659, 13123, 14010, 14013, 14071, 14291, 14309, 14322, 15010, 15168, 15173, 15499, 15938, 16009, 16039, 16247, 16586, 16835, 17088, 18605, 19190, 19336, 19600, 19622, 19725, 19729, 19730, 19732, 19735, 19737, 19739, 19741, 19744, 20138, 22110, 22202, 22213, 22245, 22418, 23731, 23918, 23919, 23924, 24445, 24781, 25167, 25169, 25170, 25171, 25172, 25173, 25174, 25962, 25965, 25967, 25978, 25979, 25980, 25982, 25987, 25989, 25992, 25996, 25999, 26003, 26009, 26013, 26014, 26015, 26018, 26021, 26026, 26028, 26029, 26030, 26031, 26032, 26033, 26035, 26036, 26037, 26038, 26042, 26044, 26046, 26047, 26049, 26051, 26054, 26057, 26063, 26067, 26068, 26069, 26075, 26078, 26082, 26975, 26989, 26990, 26993, 26996, 27006, 27016, 27017, 27022, 27025, 27026, 27029, 27034, 27038, 27041, 27061, 27062, 27064, 27065, 27066, 27662, 28175, 28424, 28427, 28555, 28563, 28573, 28581, 28602, 28661, 28690, 29993, 29997, 29999, 30004, 30011, 30012, 30014, 30016, 30021, 30026, 30029, 30040, 30041, 30042, 30043, 30044, 30045, 30047, 30048, 30051, 30052, 30053, 30054, 30058, 30063, 30067, 30081, 30082, 30084, 30086, 30087, 30088, 30090, 30102, 30103, 30115, 30118, 30119, 30120, 30122, 30124, 30126, 30131, 30133, 30136, 30137, 30147, 30150, 30152, 30200, 30201, 30233, 30241, 30243, 30699, 31759, 31833, 32007, 32384, 32386, 32550, 32551, 32565, 32908, 32909, 32910, 32915, 32920, 32954, 33834, 33916, 33922, 33954, 35360, 35398, 35406, 35424, 35439, 35440, 35519, 35851, 35867, 36386, 36536, 38092, 38320, 39007, 39538, 39539, 39541, 39551, 39567, 39572, 39576, 39594, 39612, 39625, 39630, 39631, 39640, 39641, 39643, 39644, 39645, 39652, 39653, 39655, 39656, 39657, 39658, 39661, 39665, 39666, 39691, 39726, 39732, 39733, 39734, 39737, 39738, 39741, 39742, 40168, 40719, 41196, 41197, 41200, 41205, 41223, 41667, 41693, 42032, 42662, 42720, 43051, 43113, 43461, 44479, 45068, 45070, 45553, 46084, 46436, 46439, 46443, 46451, 46454, 46472, 47900, 48420, 48549, 48568, 48577, 48632, 49382, 49954]

def read_file(data_filename="", key_filename=""):
    dataf = open(data_filename,"r")
    keyf = open(key_filename,"r")
    #read data
    lines = dataf.readlines()
    recall = float(lines[-1].replace("recall:","").replace("\n","").strip())
    precision = float(lines[-2].replace("precision:","").replace("\n","").strip())
    no = float(lines[-3].replace(" no accuracy: ","").replace("\n","").strip())
    yes = float(lines[-4].replace(" yes accuracy: ","").replace("\n","").strip())
    accuracy = float(lines[-5].replace(" Accuracy: ","").replace("\n","").strip())
    no_tokens = [float(x) for x in lines[-8].replace("[","").replace("]","").replace("(","").replace(")","").replace("tensor","").split(', ')]
    yes_tokens = [float(x) for x in lines[-10].replace("[","").replace("]","").replace("(","").replace(")","").replace("tensor","").split(', ')]
    #read labels
    key_lines = keyf.readlines()
    labels = [int('<Y>' in key) for key in key_lines]
    print("Length Match: "+str((len(labels), len(yes_tokens), len(no_tokens))))
    check_accuracy = np.sum([labels[i]==int((yes_tokens[i]>=no_tokens[i])) for i in range(len(labels))])/len(labels)
    print("Accuracy Match: "+str((accuracy, check_accuracy)))
    dataf.close()
    keyf.close()
    return accuracy, yes, no, precision, recall, yes_tokens, no_tokens, labels


def calibration(results, n=10000, title="Calibration Plot Static Model", fn="save.pdf", temp=False):
    y_true = np.array(results[7])
    y_pred = np.array(results[5])
    prob_true = []
    prob_pred = []
    for i in range(1000):
        bootstrap_idx = np.random.choice(range(len(y_true)), n)
        b_y_true = y_true[bootstrap_idx]
        b_y_pred = y_pred[bootstrap_idx]
        b_prob_true, b_prob_pred = calibration_curve(b_y_true, b_y_pred, n_bins=10)
        prob_true.append(b_prob_true)
        prob_pred.append(b_prob_pred)
    mean_x = np.mean(prob_pred, 0)
    mean_y = np.mean(prob_true, 0)
    low_y = np.mean(prob_true, 0)-np.std(prob_true, 0)
    high_y = np.mean(prob_true, 0)+np.std(prob_true, 0)
    return mean_x, mean_y, low_y, high_y

def plot(list_vals, list_key, colors, fn="temp.pdf"):
    plt.figure(figsize=(9, 5))
    plt.xlabel("Mean Predicted Probability", fontsize=22)
    plt.ylabel("Fraction of positives", fontsize=22)
    for li in range(len(list_vals)):
        plt.plot(list_vals[li][0],list_vals[li][1], color=colors[li], label=list_key[li])
        plt.fill_between(list_vals[li][0],list_vals[li][2],list_vals[li][3], color=colors[li], alpha=0.4)
    plt.plot([0, 1], [0,1], color="black", linestyle="dashed", label="Ideal Calibration")
    plt.legend(fontsize=20)
    plt.savefig(fn)
    plt.close()


calib_list = []
calib_keys = []
dynamic_results = read_file(data_filename="student_models_results/french_lmkt.txt", key_filename="french_test_key")
calib_list.append(calibration(dynamic_results, fn="student_models_results/french_lmkt.pdf"))
calib_keys.append("French LM-KT")


dynamic_results = read_file(data_filename="student_models_results/spanish_lmkt.txt", key_filename="spanish_test_key")
calib_list.append(calibration(dynamic_results, fn="student_models_results/spanish_lmkt.pdf"))
calib_keys.append("Spanish LM-KT")


plot(calib_list, calib_keys, colors=["mediumvioletred", "cornflowerblue", "orange"], fn="final_calibration_temp.pdf")
   

