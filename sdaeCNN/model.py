import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sdaeCNN.dae import DenoisingAutoencoder
from sdaeCNN.dae import DAE
from sdaeCNN.sdae import StackedDenoisingAutoEncoder

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()

    def forward(self,y_pred,y_true):
        mse_loss = nn.MSELoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        mse=mse_loss(y_pred,y_true)
        # kl = float(kl_loss(F.log_softmax(y_pred,dim=1),F.softmax(y_true,dim=1)).item())
        # kl = float(kl_loss(y_pred,y_true).item())
        psnr = 1/(1+10*torch.log10(255/torch.sqrt(mse_loss(y_pred,y_true))))
        return mse+psnr

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss,self).__init__()

    def forward(self,y_pred,y_true):
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_pred, y_true)

        # nonzero_mask = y_true != 0
        # zero_pred_mask = y_pred == 0
        # additional_loss = torch.sum(nonzero_mask & zero_pred_mask).float()

        # total_loss = mse + additional_loss
        total_loss = mse*10000
        return total_loss



def train(
    dataset: torch.utils.data.Dataset,
    autoencoder: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    name: str = "",
) -> None:
    """
    Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.

    :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
    :param autoencoder: autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
    :param update_callback: optional function of loss and validation loss to update
    :param num_workers: optional number of workers to use for data loading
    :param epoch_callback: optional function of epoch and model
    :param name : name of encoder
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers is not None else 0,
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
        )
    else:
        validation_loader = None
    loss_function = CustomMSELoss()
    # loss_function = nn.MSELoss()
    # loss_function = CustomLoss()

    autoencoder.train()
    validation_loss_value = -1
    loss_value = 0
    folder_path = "./model/sub_encoder_index_"
    file_extension = ".pth"
    save_interval =20
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            # run the batch through the autoencoder and obtain the output
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
                # mask = torch.rand(batch.size()) <= corruption
                # batch[mask] = 0.;
                # output = autoencoder(batch)
            else:
                output = autoencoder(batch)
            loss = loss_function(output, batch)
            # accuracy = pretrain_accuracy(output, batch)
            if loss_value:
                loss_value = (loss_value + float(loss.item()))/2
            else:
                loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()

            # if name == "1":
            #     print("print  grads")

                # for pname, param in autoencoder.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print("-->name", pname)
                #         print("-->para", param)
                #         print("-->grad requires", param.requires_grad)
                #         print("-->param value", param.grad)
                #         a =0

                    # grads[name] = param.requires_grad


            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
            )
        if update_freq is not None and epoch % update_freq == 0:
            if validation_loader is not None:
                validation_output = predict1(
                    validation,
                    autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=True,
                    encode=False,
                )
                # validation_inputs = []
                count = 0
                validation_loss = 0.0
                for val_batch in validation_loader:
                    validation_inputs = []
                    if (
                        isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        validation_inputs.append(val_batch[0])
                    else:
                        validation_inputs.append(val_batch)

                    val_out = validation_output[count]
                    val_in = validation_inputs[0]
                    if cuda:
                        val_in = val_in.cuda(non_blocking=True)
                        val_out = val_out.cuda(non_blocking=True)
                    loss = loss_function(val_out, val_in)
                    validation_loss = validation_loss + float(loss.item())
                    count += 1
                    # print("---------------->validation_loss",validation_loss)

                validation_loss_value = validation_loss/(count+1)
                # validation_actual = torch.cat(validation_inputs)
                # if cuda:
                #     validation_actual = validation_actual.cuda(non_blocking=True)
                #     validation_output = validation_output.cuda(non_blocking=True)
                # validation_loss = loss_function(validation_output, validation_actual)
                # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                # validation_loss_value = float(validation_loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.6f" % loss_value,
                    vls="%.6f" % validation_loss_value,
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                # validation_accuracy = -1
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                )
            if update_callback is not None:
                update_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                    validation_loss_value,
                )
        if epoch % save_interval == 0:
            file_path = f"{folder_path}{name}_{epoch}{file_extension}"
            torch.save(autoencoder.state_dict(), file_path)
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()


def pretrain(
    dataset,
    autoencoder: StackedDenoisingAutoEncoder,
    epochs: int,
    batch_size: int,
    optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
    scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    """
    Given an autoencoder, train it using the data provided in the dataset; for simplicity the accuracy is reported only
    on the training dataset. If the training dataset is a 2-tuple or list of (feature, prediction), then the prediction
    is stripped away.

    :param dataset: instance of Dataset to use for training
    :param autoencoder: instance of an autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param optimizer: function taking model and returning optimizer
    :param scheduler: function taking optimizer and returning scheduler, or None to disable
    :param validation: instance of Dataset to use for validation
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 1
    :param update_callback: function of loss and validation loss to update
    :param num_workers: optional number of workers to use for data loading
    :param epoch_callback: function of epoch and model
    :return: None
    """
    current_dataset = dataset
    current_validation = validation
    # number_of_subautoencoders = len(autoencoder.dimensions) - 1
    number_of_subautoencoders = len(autoencoder.dimensions)

    for index in range(number_of_subautoencoders):
        encoder, decoder = autoencoder.get_stack(index)

        # embedding_dimension = autoencoder.dimensions[index]
        # hidden_dimension = autoencoder.dimensions[index + 1]
        # manual override to prevent corruption for the last subautoencoder
        # if index == (number_of_subautoencoders - 1):
        # if index == (number_of_subautoencoders):
        #     corruption = None
        # initialise the subautoencoder
        sub_autoencoder = DAE(
            enCoder=encoder,
            deCoder=decoder,
            # corruption=nn.Dropout(corruption) if corruption is not None else None,
        )
        if cuda:
            sub_autoencoder = sub_autoencoder.cuda()
        ae_optimizer = optimizer(sub_autoencoder)
        ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler
        train(
            current_dataset,
            sub_autoencoder,
            epochs+index,
            batch_size,
            ae_optimizer,
            validation=current_validation,
            corruption=0.2,  # already have dropout in the DAE
            scheduler=ae_scheduler,
            cuda=cuda,
            sampler=sampler,
            silent=silent,
            update_freq=update_freq,
            update_callback=update_callback,
            num_workers=num_workers,
            epoch_callback=epoch_callback,
            name=str(index)
        )
        # copy the weights
        # sub_autoencoder.copy_weights(encoder, decoder)
        # pass the dataset through the encoder part of the subautoencoder
        # if index != (number_of_subautoencoders - 1):
        if index != (number_of_subautoencoders):

            current_dataset = TensorDataset(
                predict(
                    current_dataset,
                    sub_autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=silent,
                )
            )
            # results =[]
            # for batch in current_dataset:
            #     np.set_printoptions(threshold=math.inf)
            #     result = np.array(batch[0])
            #     results.append(result)
            # file = f"result_{epochs}_{index}.txt"
            # with open(file,'w') as f:
            #     for result in results:
            #         f.write(np.array2string(result,separator=",")+"\n")
            # data = np.array(current_dataset.tensors[0])
            # count = 0
            # for num in data:
            #     if(num == 0):
            #         count+=1
            # data1 = np.where(data == 0,1,0)
            # count = sum(sum(data1))
            if current_validation is not None:
                current_validation = TensorDataset(
                    predict(
                        current_validation,
                        sub_autoencoder,
                        batch_size,
                        cuda=cuda,
                        silent=silent,
                    )
                )
        else:
            current_dataset = None  # minor optimisation on the last subautoencoder
            current_validation = None


def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
) -> torch.Tensor:
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.

    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        # batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
            print(output)
        features.append(
            output.detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)
    # return features


def predict1(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
):
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.

    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        # batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        features.append(
            output.detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    return features


def finetune(
    dataset: torch.utils.data.Dataset,
    autoencoder: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
    """
    Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.

    :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
    :param autoencoder: autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
    :param update_callback: optional function of loss and validation loss to update
    :param num_workers: optional number of workers to use for data loading
    :param epoch_callback: optional function of epoch and model
    :return: None
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers is not None else 0,
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
        )
    else:
        validation_loader = None

    num_features = 4*4*1
    num_classes = 14
    linear_layer = nn.Linear(num_features,num_classes)

    nn.init.xavier_uniform_(linear_layer.weight.data,)
    # softmax_layer = nn.Softmax(dim=1)
    # for p in autoencoder.parameters():
    #     p.requires_grad = False
    autoencoder = nn.Sequential(
        autoencoder,
        nn.Flatten(),
        linear_layer,
        # softmax_layer
    )
    # 输出测试
    # if cuda:
    #     autoencoder.cuda()
    # for epoch in range(epochs):
    #     if scheduler is not None:
    #         scheduler.step()
    #     train_sum = 0.0
    #     data_iterator = tqdm(
    #         dataloader,
    #         leave=True,
    #         unit="batch",
    #         postfix={"epo": epoch, "lss": "%.6f" % 0.0, "train_acc": "%.6f" % 0.0, "vls": "%.6f" % -1,
    #         "val_acc": "%.6f" % 0.0},
    #         disable=silent,
    #     )
    #     for index, batch in enumerate(data_iterator):
    #         if (
    #             isinstance(batch, tuple)
    #             or isinstance(batch, list)
    #             and len(batch) in [1, 2]
    #         ):
    #             label = batch[1]
    #             batch = batch[0]
    #         if cuda:
    #             batch = batch.cuda(non_blocking=True)
    #             label = label.cuda(non_blocking=True)
    #         # run the batch through the autoencoder and obtain the output
    #         if corruption is not None:
    #             output = autoencoder(F.dropout(batch, corruption))
    #         else:
    #             output = autoencoder(batch)
    #         print(output)

    # --------------------------------------------------
    loss_function = nn.CrossEntropyLoss()
    if cuda:
        autoencoder.cuda()
    print(autoencoder)
    folder_path = "./model/classify_"
    file_extension = ".pth"
    save_interval = 20
    autoencoder.train()
    validation_loss_value = -1
    loss_value = 0.0
    validation_acc = 0.0
    train_acc = 0.0
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        train_sum = 0.0
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={"epo": epoch, "lss": "%.6f" % 0.0, "train_acc": "%.6f" % 0.0, "vls": "%.6f" % -1, "val_acc": "%.6f" % 0.0},
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                label = batch[1]
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            # run the batch through the autoencoder and obtain the output
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
            else:
                output = autoencoder(batch)

            loss = loss_function(output, label)
            tcc = acc_calculate(output, label)

            if loss_value:
                loss_value = (loss_value + float(loss.item()))/2
                train_acc = (train_acc + tcc)/2
            else:
                loss_value = float(loss.item())
                train_acc = tcc

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step(closure=None)
            data_iterator.set_postfix(
                epo=epoch, lss="%.6f" % loss_value, train_acc="%.6f" % train_acc, vls="%.6f" % validation_loss_value,
                validation_acc="%.6f" % validation_acc
            )
        if update_freq is not None and epoch % update_freq == 0:
            if validation_loader is not None:
                validation_output = predict_classify(
                    validation,
                    autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=True,
                    encode=False,
                )
                validation_inputs = []
                for val_batch in validation_loader:
                    if (
                        isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        validation_inputs.append(val_batch[1])
                    else:
                        validation_inputs.append(val_batch)
                validation_actual = torch.cat(validation_inputs)
                if cuda:
                    validation_actual = validation_actual.cuda(non_blocking=True)
                    validation_output = validation_output.cuda(non_blocking=True)


                validation_loss = loss_function(validation_output, validation_actual)
                validation_acc = acc_calculate(validation_output, validation_actual)
                validation_loss_value = float(validation_loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.6f" % loss_value,
                    train_acc="%.6f" % train_acc,
                    vls="%.6f" % validation_loss_value,
                    validation_acc="%.6f" % validation_acc
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                # validation_accuracy = -1
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                )
            if update_callback is not None:
                update_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                    validation_loss_value,
                )
        if epoch % save_interval == 0:
            file_path = f"{folder_path}{epoch}{file_extension}"
            torch.save(autoencoder.state_dict(), file_path)
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()


def predict_classify(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
) -> torch.Tensor:
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.

    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        # batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        features.append(
            output.detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)


def acc_calculate(result, target):
    result = result.argmax(axis=1)
    cmp = result.type(target.dtype) == target
    return float(cmp.type(target.dtype).sum())/len(target)
