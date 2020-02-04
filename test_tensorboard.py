from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
#     # writer.add_text('lstm', 'This is an lstm {}'.format(n_iter), n_iter)
#     # writer.add_text('rnn', 'This is an rnn {}'.format(n_iter), n_iter)
#     writer.flush()


# img = np.zeros((3, 100, 100))
# img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# img_HWC = np.zeros((100, 100, 3))
# img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
# img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# writer.add_image('my_image', img, 0)

# # If you have non-default dimension setting, set the dataformats argument.
# writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')

batch_size = 32
img_batch = np.zeros((batch_size, 3, 100, 100))
for i in range(batch_size):
    img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / batch_size * i
    img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / batch_size * i
    img_batch[i, 2] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / batch_size * i

writer = SummaryWriter()
writer.add_images('my_image_batch', img_batch, 0)


writer.close()