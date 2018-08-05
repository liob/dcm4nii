#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pydicom as dicom
import nibabel as nib
import numpy as np

from tqdm import tqdm, trange
from datetime import datetime, timedelta
from scipy.interpolate import Rbf
from PIL import Image


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

    def __repr__(self):
        return 'Series %s' % str(self.SeriesInstanceUID)

    def append(self, dcm):
        self.dicoms.append(dcm)
        InstanceNumber = int(dcm.InstanceNumber)

        # use TriggerTime if present (more accurate)
        if hasattr(dcm, 'TriggerTime'):
            SeriesTime = dcmtime2py(dcm.SeriesTime)
            TriggerTime = float(dcm.TriggerTime)
            delta_t = timedelta(milliseconds=TriggerTime)
            AcquisitionTime = SeriesTime + delta_t
        # otherwise use AcquisitionTime (resolution in s)
        else:
            AcquisitionTime = dcmtime2py(dcm.AcquisitionTime)

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
        positions = [np.array(dcm.ImagePositionPatient) for dcm in self.dicoms]
        positions.sort(key=lambda pos: np.linalg.norm(pos))
        diff = np.linalg.norm(positions[1] - positions[0])
        for i in range(1, len(positions), 1):
            current_diff = np.linalg.norm(positions[i] - positions[i-1])
            epsilon = 1e-6
            if abs(current_diff - diff) > epsilon:
                raise ValidationError('Slices not uniformly sampled')

    def _get_sorted_dicoms(self):
        """
        This function returns the dicoms of a series in a sorted list

        It assumes that the dicom images are either part of 3D Volume or a 2D+t timeseries.
        There are several ways to sort the images:
        1: Using the InstanceNumber attribute (may not be set)
        2: Using the Acquisition/Trigger Time attribute (may be equal for all images)
        3: Using the ImagePositionPatient attribute (may also be equal for all images)
        This function tries to find the best method
        """

        # Acquire all attributes
        instance_numbers = [dcm.InstanceNumber for dcm in self.dicoms]
        acquisition_times = []
        for dcm in self.dicoms:
            if hasattr(dcm, 'TriggerTime'):
                series_time = dcmtime2py(dcm.SeriesTime)
                trigger_time = float(dcm.TriggerTime)
                delta_t = timedelta(milliseconds=trigger_time)
                acquisition_time = series_time + delta_t
                # otherwise use AcquisitionTime (resolution in s)
            else:
                acquisition_time = dcmtime2py(dcm.AcquisitionTime)
            acquisition_times.append(acquisition_time)
        image_position_patients = [dcm.ImagePositionPatient for dcm in self.dicoms]

        # Check if any of these attributes are applicable for sorting the dicoms
        # Instance Numbers (must be unique and not None)
        if all(number is not None for number in instance_numbers) and \
                len(instance_numbers) == len(set(instance_numbers)):
            return sorted(self.dicoms, key=lambda dcm: dcm.InstanceNumber)

        # Acquisition times (must be unique)
        elif len(acquisition_times) == len(set(acquisition_times)):
            return [dcm for _, dcm in sorted(zip(acquisition_times, self.dicoms))]

        # Positions (must be unique)
        elif len(image_position_patients) == len(set(image_position_patients)):
            return sorted(self.dicoms, key=lambda dcm: np.linalg.norm(dcm.ImagePositionPatient))

        else:
            raise ValidationError("No sorting method applicable")

    def get_nii(self, ignore_checks=False):
        if not ignore_checks:
            self.check_integrity()

        # Build image
        shape = self.get_shape()
        n_slices = shape[2] * shape[3]
        im = np.empty([shape[0], shape[1], n_slices], dtype=np.float)

        # Iterate over all dicom images and sort them by their position
        sorted_dicoms = self._get_sorted_dicoms()
        for index, dcm in enumerate(sorted_dicoms):
            im[:, :, index] = dcm.pixel_array

        # build the affine transform
        # http://nipy.org/nibabel/dicom/dicom_orientation.html
        timesteps = sorted(self.timesteps.keys())
        tdelta = (timesteps[-1] - timesteps[0]).total_seconds() / (shape[3] - 1) * 1000  # in milliseconds

        dr, dc = float(self.dicoms[0].PixelSpacing[0]), float(self.dicoms[0].PixelSpacing[1])
        F_12, F_22, F_32, F_11, F_21, F_31 = [float(x) for x in dcm.ImageOrientationPatient]

        T1 = [float(x) for x in sorted_dicoms[0].ImagePositionPatient]

        # In case all images have same ImagePositionPatient attributes, we need to calculate the last one
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
        # nii.header['pixdim'][4] = tdelta
        # nii.header.set_xyzt_units('mm', 'sec')
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


if __name__ == "__main__":
    import argparse

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

    for input_folder in tqdm(args.input_folders):
        container = DCMContainer(use_protocol_name=args.pn)
        print('reading files for', input_folder)
        for root, dirnames, filenames in os.walk(input_folder):
            for filename in tqdm(filenames, desc='reading headers'):
                fn = os.path.join(root, filename)
                try:
                    dcm = dicom.read_file(fn, defer_size='1 KB')
                except:
                    print('%s is not a valid dicom file!' % fn)
                    continue
                container.append(dcm)

        success = False
        for p_idx, patient in container.patients.items():
            print(patient)
            for s_idx, study in patient.studies.items():
                print('    ', study)
                for se_idx, series in study.series.items():
                    shape = series.get_shape()
                    print('        ', series, shape, series.SeriesDescription)
                    suffix = ''
                    if args.interpolate:
                        suffix += '_INTERPOLATED=%ss' % str(args.stepsize)
                    fname = '%s_%s_%s_%s%s.nii' % (patient.PatientID,
                                                   study.StudyDate.strftime('%Y%m%d'),
                                                   series.SeriesNumber,
                                                   series.SeriesDescription,
                                                   suffix)
                    fname = os.path.join(args.o, fname)
                    # do not overwrite existing files
                    if os.path.isfile(fname):
                        print('%s already exists - SKIPPING' % fname)
                        continue
                    try:
                        if args.interpolate:
                            nii = series.get_nii_interprolated(args.stepsize)
                        else:
                            nii = series.get_nii()
                    except ValidationError as e:
                        print('Validation FAILED: %s' % str(e))
                        continue
                    nii.to_filename(fname)
                    print('created', fname)
                    success = True

        if not success:
            print('found no valid volumes in %s' % input_folder)