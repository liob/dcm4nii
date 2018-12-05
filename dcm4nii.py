# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import pydicom as dicom
import nibabel as nib
import numpy as np

import argparse
import traceback

from tqdm import tqdm, trange
from datetime import datetime, timedelta
from scipy.interpolate import Rbf


def dcmtime2py(t):
    try:
        return datetime.strptime(t, '%H%M%S')
    except:
        pass
    try:
        return datetime.strptime(t, '%H%M%S.%f')
    except:
        raise


def dcmdate2py(t):
    return datetime.strptime(t, '%Y%m%d')


class ValidationError(Exception):
    pass


class DCMContainer(object):
    def __init__(self, use_protocol_name=False):
        self.patients = {}
        self.use_protocol_name = use_protocol_name

    def append(self, dcm):
        if isinstance(dcm, dicom.dicomdir.DicomDir):
            # do not process dicom directory files
            return
        if not dcm.PatientID in self.patients:
            self.patients[dcm.PatientID] = Patient(PatientID=dcm.PatientID, container=self)
        self.patients[dcm.PatientID].append(dcm)


class Patient(object):
    def __init__(self, PatientID, container):
        self.PatientID = PatientID
        self.container = container
        self.studies = {}

    def __repr__(self):
        return 'Patient %s' % str(self.PatientID)

    def __eq__(self, PatientID):
        return self.PatientID == PatientID

    def __ne__(self, PatientID):
        return self.PatientID != PatientID

    def append(self, dcm):
        if not dcm.StudyInstanceUID in self.studies:
            self.studies[dcm.StudyInstanceUID] = Study(StudyInstanceUID=dcm.StudyInstanceUID,
                                                       patient=self)
        self.studies[dcm.StudyInstanceUID].append(dcm)


class Study(object):
    def __init__(self, StudyInstanceUID, patient):
        self.StudyInstanceUID = StudyInstanceUID
        self.patient = patient
        self.series = {}
        self.StudyDate = None
        self.log = ''
        self.AccessionNumber = ''

    def __repr__(self):
        return 'Study %s' % str(self.StudyInstanceUID)

    def append(self, dcm):
        # implement option to use ProtocolName instead of SeriesInstanceUID to identify series
        if self.patient.container.use_protocol_name:
            if hasattr(dcm, 'ProtocolName'):
                series_identifier = dcm.ProtocolName
            else:
                series_identifier = dcm.SeriesInstanceUID
        else:
            series_identifier = dcm.SeriesInstanceUID

        if not series_identifier in self.series:
            self.series[series_identifier] = Series(SeriesInstanceUID=dcm.SeriesInstanceUID,
                                                    patient=self.patient,
                                                    study=self)

        self.series[series_identifier].append(dcm)
        if self.StudyDate == None:
            self.StudyDate = dcmdate2py(dcm.StudyDate)

        if self.AccessionNumber == '':
            try:
                self.AccessionNumber = dcm.AccessionNumber
            except:
                self.AccessionNumber = 'undefined'

    def set_series_filenames(self):
        all_series = []
        for series_index, series in self.series.items():
            if isinstance(series.SeriesNumber, int):
                all_series.append([series.SeriesNumber, series.SeriesDescription, '', series_index])
        # print "list sortetd by series_number"
        all_series = sorted(all_series, key=lambda x: x[0])

        for element in all_series:
            element[1] = element[1].replace('-', '_').lower()
            element[1] = element[1].replace(' ', '')
            element[1] = element[1].replace('fs_tra', 'tra_fs')

        for index in range(len(all_series)):
            # update
            if (all_series[index][1].find('navigator') >= 0 and all_series[index][1].find('resp') >= 0) or \
                    all_series[index][1].find('mapping') >= 0 or all_series[index][1].find('mip_cor') >= 0 or \
                    all_series[index][1].find('resp') >= 0 or all_series[index][1].find('ringe3phasen') >= 0 or \
                    all_series[index][1].find('4mm3messungen') >= 0 or all_series[index][1].find('localizer') >= 0 or \
                    all_series[index][1].find('carebolus') >= 0 or all_series[index][1] == '':
                all_series[index][2] = 'error_not_a_valid_imagename'

        # 1 t1fl2d
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_fl2d_tra_5mm') >= 0:
                    all_series[index][2] = 't1_fl2d_5mm'
                    break

        #  2 t2_spc_cor mrcp use the last one
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                # update
                if all_series[index][1].find('mrcp') >= 0:
                    all_series[index][2] = 't2_spc_cor'
                    break  # TODO or continue to find all other mrcp?

        # update
        # 3 dixon
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                if all_series[index][1].find('dixon') >= 0:
                    for i in range(4):
                        # TODO correct names
                        if all_series[index - i][1].find('opp') >= 0:
                            all_series[index - i][2] = 't1_vibe_dixon_opp'
                        elif all_series[index - i][1].find('in') >= 0:
                            all_series[index - i][2] = 't1_vibe_dixon_in'
                        elif all_series[index - i][1].find('f') >= 0:
                            all_series[index - i][2] = 't1_vibe_dixon_f'
                        elif all_series[index - i][1].find('w') >= 0:
                            all_series[index - i][2] = 't1_vibe_dixon_w'
                    break

        # update
        #  ?? navigator
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                # update
                if all_series[index][1].find('navigator') >= 0:
                    all_series[index][2] = 't2_tse_tra_navigator'
                    break

        # update
        #  ?? diff
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                # update
                if all_series[index][1].find('diff') >= 0 and all_series[index][1].find('adc') >= 0:
                    all_series[index][2] = 'ep2d_diff_adc'
                    break
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                # update
                if all_series[index][1].find('diff') >= 0 and not all_series[index][1].find('adc') >= 0:
                    all_series[index][2] = 'ep2d_diff'
                    break

        # 6 t1_vibe_tra
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_tra_fs') >= 0 and all_series[index][1].find('nativ'):
                    all_series[index][2] = 't1_vibe_fs_nativ'
                    break

        # 7 arteriell_1 dyn
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_tra_dyn') >= 0 and all_series[index][1].find('art') >= 0:
                    searching = True
                    j = 0
                    while searching:
                        if all_series[index + j][1].find('t1_vibe_tra_dyn') >= 0 and all_series[index + j][1].find(
                                'art') >= 0:
                            all_series[index + j][2] = 't1_vibe_fs_dyn_art_%s_' % (j + 1)
                            j += 1
                        else:
                            searching = False
                    max = j
                    if max > 3:
                        while j > 0:
                            all_series[index + j - 1][2] = 'error_to_many_art'
                            j -= 1
                    else:
                        while j > 0:
                            all_series[index + j - 1][2] += str(max)
                            j -= 1
                    break

        # 8 ven_1 4mm
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_tra_fs_dyn_4') >= 0:
                    searching = True
                    j = 0
                    while searching:
                        if all_series[index + j][1].find('t1_vibe_tra_fs_dyn_4') >= 0:
                            all_series[index + j][2] = 't1_vibe_fs_dyn_ven_4mm_%s_' % (j + 1)
                            j += 1
                        else:
                            searching = False
                    max = j
                    if max > 3:
                        while j > 0:
                            all_series[index + j - 1][2] = 'error_to_many_ven'
                            j -= 1
                    else:
                        while j > 0:
                            all_series[index + j - 1][2] += str(max)
                            j -= 1
                    break

        # 8 ven_2 2mm
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_tra_fs_dyn_2') >= 0:
                    searching = True
                    j = 0
                    while searching:
                        if all_series[index + j][1].find('t1_vibe_tra_fs_dyn_2') >= 0:
                            all_series[index + j][2] = 't1_vibe_fs_dyn_ven_2mm_%s_' % (j + 1)
                            j += 1
                        else:
                            searching = False
                    max = j
                    if max > 3:
                        while j > 0:
                            all_series[index + j - 1][2] = 'error_to_many_ven'
                            j -= 1
                    else:
                        while j > 0:
                            all_series[index + j - 1][2] += str(max)
                            j -= 1
                    break

        # 12 / 13 t1_vibe_tra_fs_late
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_tra_fs') >= 0:
                    all_series[index][2] = 't1_vibe_tra_fs_late'
                    break

        # 12 / 14 t1_vibe_cor_fs_late
        for index in reversed(range(len(all_series))):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_cor_fs') >= 0:
                    all_series[index][2] = 't1_vibe_cor_fs_late'
                    break

        # 9 t1_vibe_cor_fs
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t1_vibe_cor_fs') >= 0:
                    all_series[index][2] = 't1_vibe_cor_fs'
                    break

        # 11 t2_haste_fs_cor
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t2_haste_fs_cor') >= 0 or \
                        all_series[index][1].find('t2_haste_cor_fs') >= 0:
                    all_series[index][2] = 't2_haste_fs_cor'
                    break

        # ?? t2_tse_tra
        for index in range(len(all_series)):
            if all_series[index][2] == '':
                if all_series[index][1].find('t2_tse_tra_fs') >= 0 or \
                        all_series[index][1].find('t2_tse_fs_tra') >= 0:
                    all_series[index][2] = 't2_tse_tra_fs'
                    break

        for current_series in all_series:
            if current_series[2] == '':
                # update
                # move to subfolder
                current_series[2] = 'backupfile_' + str(current_series[0]) + '_' + str(current_series[1])

        self.log = ''
        for current_series in all_series:
            self.log += str(current_series[0]) + ':' + str(current_series[1]) + ' -> ' + str(current_series[2]) + '\n'
            pass

        for current_series in all_series:
            self.series[current_series[3]].filename = current_series[2]

        # for s in all_series:
        # print(s[1])


class Series(object):
    def __init__(self, SeriesInstanceUID, patient, study):
        self.SeriesInstanceUID = SeriesInstanceUID
        self.patient = patient
        self.study = study
        self.ProtocolName = ''
        self.SeriesDescription = ''
        self.SeriesNumber = ''
        self.dicoms = []
        self.timesteps = {}
        self.dcm_first = None
        self.dcm_last = None
        self.ContrastBolusAgent = ''
        self.SeriesTime = None
        self.FolderName = ''
        self.Phase = ''
        self.filename = ''

    def __repr__(self):
        return 'Series %s' % str(self.SeriesInstanceUID)

    def append(self, dcm):
        self.dicoms.append(dcm)
        InstanceNumber = int(dcm.InstanceNumber)

        self.SeriesTime = dcmtime2py(dcm.SeriesTime)

        if self.SeriesTime == None:
            self.SeriesTime = dcmdate2py(dcm.SeriesTime)

        # use TriggerTime if present (more accurate)
        if hasattr(dcm, 'TriggerTime'):
            SeriesTime = dcmtime2py(dcm.SeriesTime)
            TriggerTime = float(dcm.TriggerTime)
            delta_t = timedelta(milliseconds=TriggerTime)
            AcquisitionTime = SeriesTime + delta_t
        # otherwise use AcquisitionTime (resolution in s)
        elif hasattr(dcm, 'AcquisitionTime'):
            AcquisitionTime = dcmtime2py(dcm.AcquisitionTime)
        else:
            raise ValidationError("Dicom file has neither Trigger nor Acquisition time")

        if not AcquisitionTime in self.timesteps:
            self.timesteps[AcquisitionTime] = {}
        self.timesteps[AcquisitionTime][InstanceNumber] = dcm

        if self.dcm_first == None:
            self.dcm_first = dcm
        else:
            if InstanceNumber < int(self.dcm_first.InstanceNumber):
                self.dcm_first = dcm

        if self.dcm_last == None:
            self.dcm_last = dcm
        else:
            if InstanceNumber > int(self.dcm_last.InstanceNumber):
                self.dcm_last = dcm

        if self.ProtocolName == '':
            try:
                self.ProtocolName = dcm.ProtocolName
            except:
                self.ProtocolName = 'undefined'

        if self.SeriesDescription == '':
            try:
                self.SeriesDescription = dcm.SeriesDescription
            except:
                self.SeriesDescription = 'undefined'

        if self.FolderName == '':
            if self.SeriesDescription.find('_3DMRCP') >= 0:
                orientation = '3DMRCP'
            elif self.SeriesDescription.find('t1_') >= 0:
                orientation = 't1'
            elif self.SeriesDescription.find('t2_') >= 0:
                orientation = 't2'
            else:
                orientation = 'undef'
            self.FolderName = orientation

        if self.ContrastBolusAgent == '':
            try:
                self.ContrastBolusAgent = dcm.ContrastBolusAgent
                self.FolderName += '_%s' % dcm.ContrastBolusAgent
            except:
                self.ContrastBolusAgent = 'undefined'
                self.FolderName += '_noContrast'

        if self.SeriesNumber == '':
            try:
                self.SeriesNumber = dcm.SeriesNumber
            except:
                self.SeriesNumber = 'undefined'

    def get_shape(self):
        dcm = self.dicoms[0]
        shape = dcm.pixel_array.shape
        n_slices = len(self.timesteps[next(iter(self.timesteps))])
        return shape[0], shape[1], n_slices, len(self.timesteps)

    def check_integrity(self, max_t_delta=1.5, c_delta_t=True):
        # check all timepoints have the same number of observations
        observations = []
        for timestep in sorted(self.timesteps.keys()):
            observations.append(len(self.timesteps[timestep].keys()))
        observations_checked = [x == observations[0] for x in observations]
        if not sum(observations_checked) == len(observations):
            raise ValidationError('inconsistent number of DCMs per timestep! Files missing?')

        # check no major variance is in the distribution of observations regarding time
        timesteps = sorted(self.timesteps.keys())
        t_gradient = []
        for i in range(len(timesteps)):
            if i == 0:
                continue
            else:
                t_delta = (timesteps[i] - timesteps[i - 1]).total_seconds()
                t_gradient.append(t_delta)
        t_checked = [x > max_t_delta for x in t_gradient]
        if (sum(t_checked) > 0) & c_delta_t:
            raise ValidationError('Major variance in timesteps. t_gradient: %s' % str(t_gradient))

        if not hasattr(self.dcm_first, 'ImageOrientationPatient'):
            raise ValidationError('No Ímage Orientation Patient available')

        # We can only create nifti files, if all orientations within a series are the same
        orientation = self.dcm_first.ImageOrientationPatient
        for dcm in self.dicoms:
            if dcm.ImageOrientationPatient != orientation:
                raise ValidationError('Varying image orientation in series')

        # For now, we will assume that we have either one time step with many slices or many slices with one timestep
        # -> shape(X,Y,n_slices, 1) or shape(X;Y,1,n_time_steps)
        if self.get_shape()[2] != 1 and self.get_shape()[3] != 1:
            raise ValidationError('Invalid shape: ', self.get_shape())

        # We also need to ensure that the slices are uniformly sampled
        # ToDo: Maybe add get_nii_interpolated to functionality?
        sorted_dicoms = self._get_sorted_dicoms()
        positions = [np.array(dcm.ImagePositionPatient) for dcm in sorted_dicoms]
        # Look at the deviation of neighbour slice distances
        diffs = []
        for i in range(1, len(positions), 1):
            diffs.append(np.linalg.norm(positions[i] - positions[i - 1]))
        dev = np.std(diffs)
        epsilon = 1e-5
        if dev > epsilon:
            raise ValidationError("Slices not uniformly sampled, std deviation: ", dev)

    def _get_sorted_dicoms(self):
        """
        This function returns the dicoms of this series in a sorted list by applying the method described in
        https://itk.org/pipermail/insight-users/2003-September/004762.html
        It assumes that the dicom images are either part of 3D Volume or a 2D+t timeseries.
        """

        # Calculate normal of slices
        row_cosines = self.dicoms[0].ImageOrientationPatient[:3]
        col_cosines = self.dicoms[0].ImageOrientationPatient[3:]
        n = np.cross(row_cosines, col_cosines)

        # Calculate dist value for all slices and order dicoms by that
        distances = [np.dot(n, dcm.ImagePositionPatient) for dcm in self.dicoms]
        result = [x for x, _ in sorted(zip(self.dicoms, distances), key=lambda x: x[1])]
        return result

    def get_nii(self, ignore_checks=False):
        if not ignore_checks:
            self.check_integrity()

        # Build image
        shape = self.get_shape()
        n_slices = shape[2] * shape[3]
        im = np.empty([shape[0], shape[1], n_slices], dtype=np.float)

        # Iterate over all dicom images
        sorted_dicoms = self._get_sorted_dicoms()
        for index, dcm in enumerate(sorted_dicoms):
            im[:, :, index] = dcm.pixel_array
        im = im[:, ::-1, ::-1]

        # Build the affine transform
        # http://nipy.org/nibabel/dicom/dicom_orientation.html
        dr, dc = float(self.dicoms[0].PixelSpacing[0]), float(self.dicoms[0].PixelSpacing[1])
        F_12, F_22, F_32, F_11, F_21, F_31 = [float(x) for x in dcm.ImageOrientationPatient]

        T1 = [float(x) for x in sorted_dicoms[0].ImagePositionPatient]

        # In case all images have same ImagePositionPatient attributes, we need to calculate the last one (TN)
        # ToDO: Add implementation. For now raise Error. No Idea how to get the right directional vector.
        # ToDo: Cross product is no not unique solution
        if sorted_dicoms[0].ImagePositionPatient == sorted_dicoms[-1].ImagePositionPatient:
            raise ValidationError("All ImagePositionPatient attributes equal")

        TN = [float(x) for x in sorted_dicoms[-1].ImagePositionPatient]
        N = im.shape[2]

        affine = [[F_11 * dr, F_12 * dc, (T1[0] - TN[0]) / (1 - N), T1[0]],
                  [F_21 * dr, F_22 * dc, (T1[1] - TN[1]) / (1 - N), T1[1]],
                  [F_31 * dr, F_32 * dc, (T1[2] - TN[2]) / (1 - N), T1[2]],
                  [0, 0, 0, 1]]

        affine = np.array(affine)
        nii = nib.Nifti1Image(im, affine)
        return nii

    def get_nii_interprolated(self, stepsize):
        # def interpolate(x, y, _x, function='gaussian', stepsize=stepsize, smooth=0.01):
        def interpolate(x, y, _x, function='linear', stepsize=stepsize, smooth=0):
            rbf = Rbf(x, y, function=function, smooth=smooth)
            _y = rbf(_x)
            return _y

        self.check_integrity(c_delta_t=False)
        img = self.get_nii(ignore_checks=True)
        timesteps = sorted(self.timesteps.keys())
        start_time = timesteps[0]
        x = [(x - start_time).total_seconds() for x in timesteps]
        _x = np.arange(x[-1], step=stepsize)

        im = img.get_data()
        shape = im.shape
        im = im.reshape((np.prod(shape[:-1]), shape[-1]))

        target_shape = shape[:-1] + (len(_x),)
        interpolated_im = np.empty((im.shape[0], len(_x)), dtype=np.float)
        for idx in trange(im.shape[0], desc='interpolating'):
            interpolated_im[idx] = interpolate(x, im[idx], _x)
        interpolated_im = interpolated_im.reshape(target_shape)

        nii = nib.Nifti1Image(interpolated_im, img.affine)
        nii.header['pixdim'][4] = stepsize * 1000
        nii.header.set_xyzt_units('mm', 'sec')
        return nii


def retrieve_series_names():
    input_folder = "dicoms_loop_15_18"

    print('reading files for', input_folder)

    directories = [os.path.join(input_folder, d) for d in os.listdir(input_folder)
                   if os.path.isdir(os.path.join(input_folder, d))]

    result = []
    for index, directory in enumerate(directories):
        print(index, "of ", len(directories))

        for root, dirnames, filenames in os.walk(directory):
            container = DCMContainer(use_protocol_name=False)
            for filename in filenames:
                fn = os.path.join(root, filename)
                try:
                    dcm = dicom.read_file(fn, defer_size='1 KB')
                    container.append(dcm)
                except:
                    #print('%s is not a valid dicom file!' % fn)
                    continue

            for p_idx, patient in container.patients.items():
                # print(patient)
                for s_idx, study in patient.studies.items():
                    series = [s for _, s in study.series.items()]
                    series = sorted(series, key=lambda x: x.SeriesNumber)

                    descriptions = [s.SeriesDescription for s in series]
                    result.append(descriptions)

            tmp = np.array(result)
            np.save("tmp.np", tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert dicom files to nifti (2D/3D/4D)')
    parser.add_argument('-o', metavar='OUTPUT DIR', type=str, default='.',
                        help='output directory for the plots')
    parser.add_argument('--pn', dest='pn', default=False, action='store_true',
                        help='use Protocol Name instead of StudyInstanceUID to join series')
    parser.add_argument('--interpolate', dest='interpolate', default=False, action='store_true',
                        help='enable temporal interpolation')
    parser.add_argument('--stepsize', metavar='STEPSIZE', type=float, default=1.5,
                        help='temporal resolution for interpolation in seconds')
    parser.add_argument('input_folders', metavar='FOLDER', type=str, nargs='+',
                        help='input folder')
    args = parser.parse_args()

    for input_folder in args.input_folders:

        directories = [os.path.join(input_folder, directory) for directory in os.listdir(input_folder)
                       if os.path.isdir(os.path.join(input_folder, directory))]
        for directory in tqdm(directories):
            container = DCMContainer(use_protocol_name=args.pn)

            filenames = os.listdir(directory)
            for filename in filenames:
                fn = os.path.join(directory, filename)
                try:
                    dcm = dicom.read_file(fn, defer_size='1 KB')
                    container.append(dcm)
                except:
                    continue

            for p_idx, patient in container.patients.items():
                for s_idx, study in patient.studies.items():
                    folder_name = os.path.join(args.o, study.AccessionNumber + "_" + str(study.StudyDate.date()))
                    for se_idx, series in study.series.items():
                        file_name = os.path.join(folder_name, str(series.SeriesNumber) + "_" + series.SeriesDescription + ".nii.gz")
                        try:
                            if not os.path.exists(folder_name):
                                os.makedirs(folder_name)
                            nii = series.get_nii()
                            nii.to_filename(file_name)

                        except Exception as e:
                            error_log_path = os.path.join(folder_name, "error_log")
                            with open(error_log_path, "a+") as f:
                                f.writelines("Exception for " + str(series.SeriesNumber) + "_" + series.SeriesDescription)
                                f.writelines(traceback.format_exc())
                                f.writelines("\n\n")


