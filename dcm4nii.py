# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pydicom as dicom
import nibabel as nib
import numpy as np

import argparse
import traceback

from tqdm import tqdm
from datetime import datetime


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
        if self.StudyDate is None:
            self.StudyDate = dcmdate2py(dcm.StudyDate)

        if self.AccessionNumber == '':
            try:
                self.AccessionNumber = dcm.AccessionNumber
            except:
                self.AccessionNumber = 'undefined'


class Series(object):
    def __init__(self, SeriesInstanceUID, patient, study):
        self.SeriesInstanceUID = SeriesInstanceUID
        self.patient = patient
        self.study = study
        self.ProtocolName = ''
        self.SeriesDescription = ''
        self.SeriesNumber = ''
        self.dicoms = []
        self.ContrastBolusAgent = ''
        self.SeriesTime = None
        self.FolderName = ''
        self.Phase = ''
        self.filename = ''

    def __repr__(self):
        return 'Series %s' % str(self.SeriesInstanceUID)

    def append(self, dcm):
        self.dicoms.append(dcm)
        self.SeriesTime = dcmtime2py(dcm.SeriesTime)

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
        return self.dicoms[0].pixel_array.shape[0], self.dicoms[0].pixel_array.shape[1], len(self.dicoms)

    def check_integrity(self, max_t_delta=1.5, c_delta_t=True):
        # check no major variance is in the distribution of observations regarding time
        timesteps = sorted(dcmtime2py(dcm.AcquisitionTime) for dcm in self.dicoms)
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

        if not hasattr(self.dicoms[0], 'ImageOrientationPatient'):
            raise ValidationError('No Ímage Orientation Patient available')

        # We can only create nifti files, if all orientations within a series are the same
        orientation = self.dicoms[0].ImageOrientationPatient
        for dcm in self.dicoms:
            if dcm.ImageOrientationPatient != orientation:
                raise ValidationError('Varying image orientation in series')

        # We also need to ensure that the slices are uniformly sampled
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
        im = np.empty(self.get_shape(), dtype=np.float)

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
                except Exception as e:
                    continue

            for p_idx, patient in container.patients.items():
                for s_idx, study in patient.studies.items():
                    folder_name = os.path.join(args.o, study.AccessionNumber + "_" + str(study.StudyDate.date()))
                    for se_idx, series in study.series.items():
                        file_name = os.path.join(folder_name,
                                                 str(series.SeriesNumber) + "_" + series.SeriesDescription + ".nii.gz")
                        try:
                            if not os.path.exists(folder_name):
                                os.makedirs(folder_name)
                            nii = series.get_nii()
                            nii.to_filename(file_name)

                        except Exception as e:
                            error_log_path = os.path.join(folder_name, "error_log")
                            with open(error_log_path, "a+") as f:
                                f.writelines("Exception for " + str(series.SeriesNumber) + "_"
                                             + series.SeriesDescription)
                                f.writelines(traceback.format_exc())
                                f.writelines("\n\n")


