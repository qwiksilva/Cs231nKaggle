import matplotlib.pyplot as plt

#f = open('/data/run2/loss.txt')
f = open('loss.txt')

i = 0
systole_loss_rmse_val = []
systole_loss_rmse_train = []

diastole_loss_rmse_val = []
diastole_loss_rmse_train = []

crps_val = []
crps_train = []

for line in f:
	i += 1
	if i == 1:
		continue
	line = line.split()
	systole_loss_rmse_train.append(float(line[1].strip()))
	diastole_loss_rmse_train.append(float(line[2].strip()))
	systole_loss_rmse_val.append(float(line[3].strip()))
	diastole_loss_rmse_val.append(float(line[4].strip()))
	if (i == 1) or (i%5 == 0):
		crps_train.append(float(line[5].strip()))
		crps_val.append(float(line[6].strip()))

plot.axhline(y=0.122, label = 'baseline')
plt.plot(range(1,len(systole_loss_rmse_train) + 1), systole_loss_rmse_train, 'r-')
plt.plot(range(1,len(systole_loss_rmse_val) + 1), systole_loss_rmse_val, 'r--')
plt.plot(range(1,len(diastole_loss_rmse_train) + 1), diastole_loss_rmse_train, 'b-')
plt.plot(range(1,len(diastole_loss_rmse_val) + 1), diastole_loss_rmse_val, 'b--')

plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('RMSE Loss')
plt.legend(['systole train loss', 'systole validation loss', 'diastole train loss', 'diastole validation loss'])
plt.show()

plt.plot(range(1,len(crps_train) + 1), crps_train, 'r-')
plt.plot(range(1,len(crps_val) + 1), crps_val, 'r--')

plt.xlabel('Per 5th iteration')
plt.ylabel('CRPS')
plt.title('CRPS Loss')
plt.legend(['crps train', 'crps validation'])
plt.show()