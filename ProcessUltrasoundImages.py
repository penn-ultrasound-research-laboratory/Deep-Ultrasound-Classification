from datetime import datetime
import argparse, os

## PSEUDO-CODE

def process_patient(
    absolute_path_to_patient_folder, 
    relative_path_to_frames_folder, 
    relative_path_to_focus_output_folder,
    manifest_file_pointer,
    failure_log_file_pointer):
    '''Processes a patient folder

    Each patient's ultrasound lives in a unique folder. Expectation is that a consistently named folder exists for each patient
    containing the individual frames of the patient's ultrasound. For each patient,
    go frame by frame and process the frf = open('workfile', 'w')ame for type (color/grayscale), information (WF, RAD/ARAD, etc.), and a frame focus. The frame
    focus is clearly delineated for color frames and must be inferred for grayscale frames. 

    Arguments:
        absolute_path_to_patient_folder: absolute path to patient folder
        relative_path_to_frames_folder: relative path from the patient folder to patient frames folder
        relative_path_to_focus_output_folder: relative path from the patient folder to frame focus output folder 
        manifest_file_pointer: file pointer to the manifest
        failure_log_file_pointer: file pointer to the failure log
        patient_type_label: (optional) type of patient. Prefix in filename and present in all records

    Returns:
        A tuple containing: (Integer: #sucesses, Integer: #failures)
    '''
    return True

def process_patient_set(
    path_to_top_level_directory, 
    relative_path_to_frames_folder,
    relative_path_to_focus_output_folder,
    path_to_manifest_output_directory, 
    path_to_failures_output_directory,
    patient_type_label):
    '''Processes a set of patients from a top level directory.

    A set of patient folders lives in a top level directory. Each patient is in its own folder. This script
    will generate two files: a manifest (manifest.json) containing all of the parsed information from patient 
    ultrasounds and a failure log (failure.txt) that details any issues parsing. 

    Arguments:
        path_to_top_level_directory: absolute path to top level directory containing patient folders
        relative_path_to_frames_folder: relative path from the patient folder to patient frames folder
        relative_path_to_focus_output_folder: relative path from the patient folder to frame focus output folder 
        path_to_manifest_output_directory: absolute path to manifest output directory
        path_to_failures_output_directory: absolute path to failure log output directory
        patient_type_label: (optional) type of patient. Prefix in filename and present in all records

    Returns:
        A tuple containing: (Integer: #sucesses, Integer: #failures)
    '''


    timestamp =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    manifest_absolute_path = '{0}/manifest_{1}_{2}.json'.format(
        path_to_manifest_output_directory.rstrip('/'), 
        (patient_type_label if patient_type_label is not None else 'UNSPEC'),
        timestamp)

    failure_log_absolute_path = '{0}/failures_{1}_{2}.txt'.format(
        path_to_failures_output_directory.rstrip('/'),
        (patient_type_label if patient_type_label is not None else 'UNSPEC'),
        timestamp)

    manifest_file = open(manifest_absolute_path, 'a')
    failures_file = open(failure_log_absolute_path, 'a')

    patient_subdirectories = [name for name in os.listdir(path_to_top_level_directory) 
        if os.path.isdir(os.path.join(path_to_top_level_directory, name))]

    for patient in patient_subdirectories:

        print('Processing patient: {0}'.format(patient))

        patient_directory_absolute_path = '{0}/{1}'.format(
            path_to_top_level_directory.rstrip('/'),
            patient)

        process_patient(
            patient_directory_absolute_path, 
            relative_path_to_frames_folder, 
            relative_path_to_focus_output_folder,
            manifest_file,
            failures_file)

    manifest_file.close()
    failures_file.close()

## for each image in a sub-folder

    # Create a manifest.json and an failures.json

    ## Make an Assessment composed of Tasks. 
    # If any task fails, need a log marking the frame/folder/class combo as issue with an issue message
    # Assessment should be stored as JSON object:

    # {
    #   pathToFrame: /path_to_frame 
    #   patientId: String
    #   radiality: Enum (RAD, ARAD)
    #   colorLevel: integer (%)
    #   wallFilter: integer 
    #   pulseRepititionFrequency: integer 
    #   pathToFocus: /path_to_focus
    # }

    # Task 1) Radiality
    # Task 2) Mode determination (Grayscale/Compound-SonoCT/Color/CPA)
    
    # GRAYSCALE:
        # Task 3) Size determination in lower left

    # COLOR/CPA
        # Task 3.A) Color/CPA Level
        # Task 3.B) Pulse repitition frequency
        # Task 3.C) Wall Filter

    # Task 4) Pull the image focus out of the frame
        # GRAYSCALE
            # Grab a large cropping from the center? 
        
        # COLOR/CPA (DONE)
            # Pull the green rectangle


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_top_level_directory',
        help='absolute path to top level directory containing patient folders')
    
    parser.add_argument('-out', '--path_to_manifest_output_directory', type=str, default='.', 
        help='absolute path to manifest output directory')

    parser.add_argument('-fail', '--path_to_failures_output_directory', type=str, default='.', 
        help='absolute path to failure log output directory')
    
    parser.add_argument('-frames', '--relative_path_to_frames_directory', type=str, default='frames',
        help='relative path from the patient folder to patient frames folder')
    
    parser.add_argument('-focus', '--relative_path_to_focus_output_folder', type=str, default='focus',
        help='relative path from the patient folder to frame focus output folder ')

    parser.add_argument('-label', '--patient_type_label', type=str, default=None, 
        help='type of patient. Prefix in filename and present in all records')

    args = vars(parser.parse_args())

    process_patient_set(
        args['path_to_top_level_directory'],
        args['relative_path_to_frames_directory'],
        args['relative_path_to_focus_output_folder'],
        args['path_to_manifest_output_directory'], 
        args['path_to_failures_output_directory'],
        args['patient_type_label'])

