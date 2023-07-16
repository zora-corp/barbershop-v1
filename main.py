import argparse
import os
import shutil
from pathlib import Path
import dlib
import torchvision
import PIL

from models.Alignment import Alignment
from models.Blending import Blending
from models.Embedding import Embedding
from utils.shape_predictor import align_face
from utils.drive import open_url
import requests


def update_arguments(args, session, host, record_id):
    detail_endpoint = f"https://{host}/api/v3/gan-based-generated-styles/{record_id}"

    response = session.get(detail_endpoint)
    data = response.json()
    provided_arguments = data['programArguments']

    d = vars(args)
    for k, v in provided_arguments.iteritems():
        d[k] = v
    return data


def download_file(file_path, url):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def upload_file(file_path, file_name, mime, url, data=None, session=requests.Session()):
    files = {'upload_file': (file_name, open(file_path, 'rb'), mime)}
    response = session.put(url, files=files, data=data)
    pass


# noinspection DuplicatedCode
def align_faces_in_image(predictor, args, image_path, destination_path):
    faces = align_face(image_path, predictor)
    if len(faces) > 0:
        face = faces[0]

        if args.output_size:
            factor = 1024 // args.output_size
            assert args.output_size * factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
        face.save(Path(destination_path))


def main(args):
    host = args.api_host
    record_id = args.api_record_id
    token = args.api_access_token

    session = requests.Session()
    session.headers.update({'Auth': f"Bearer {token}"})

    # Update Arguments
    data = update_arguments(args, session, host, record_id)  # extension .png
    identity_file = f"{record_id}i.{data['identityImage']['extension']}"
    structure_file = f"{record_id}s.{data['structureImage']['extension']}"
    appearance_file = f"{record_id}a.{data['appearanceImage']['extension']}"

    # Download files
    download_file(f"unprocessed/{identity_file}", data['identityImage']['url'])
    download_file(f"unprocessed/{structure_file}", data['structureImage']['url'])
    download_file(f"unprocessed/{appearance_file}", data['appearanceImage']['url'])

    d = vars(args)

    # Align The Images
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading Shape Predictor")
    f = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir,
                 return_path=True)
    predictor = dlib.shape_predictor(f)
    dst_dir = args.input_dir

    align_faces_in_image(predictor, args, f"unprocessed/{identity_file}", f"{dst_dir}/{record_id}i.png")
    align_faces_in_image(predictor, args, f"unprocessed/{structure_file}", f"{dst_dir}/{record_id}s.png")
    align_faces_in_image(predictor, args, f"unprocessed/{appearance_file}", f"{dst_dir}/{record_id}a.png")

    # -- Update Input Parameters
    d['im_path1'] = f"{record_id}i.png"
    d['im_path2'] = f"{record_id}s.png"
    d['im_path3'] = f"{record_id}a.png"

    # Update Output Parameters
    d['output_dir'] = f"output_{record_id}"

    # RUN THE MODEL
    run_model(args)

    # Upload the results
    result_file_endpoint = f"https://{host}/api/v3/gan-based-generated-styles/{record_id}/resultFiles"
    # Upload W+ and FS Images
    for y in ['W+', 'FS']:
        for x in ['i', 's', 'a']:
            file_name = f'{y}/{record_id}{x}.png'
            file_path = os.path.join(args.output_dir, file_name)
            upload_file(file_path, file_name, 'image/png',
                        result_file_endpoint, data={'filepath': file_path},
                        session=session)
    # Upload Align image
    align_file_name = f'Align_{args.sign}/{record_id}i_{record_id}s.png'
    upload_file(os.path.join(args.output_dir, align_file_name), align_file_name, 'image/png',
                result_file_endpoint, data={'filepath': file_path},
                session=session)

    # Upload geometry
    # ---

    # Upload result image
    save_dir = os.path.join(args.output_dir, 'Blend_{}'.format(args.sign))
    image_path = os.path.join(save_dir, '{}_{}_{}.png'.format(identity_file, structure_file, appearance_file))

    # The final endpoint should be called at end,
    # because it'll mark the style as completed
    final_image_endpoint = f"https://{host}/api/v3/gan-based-generated-styles/{record_id}/finalImage"
    upload_file(image_path, f'{identity_file}_{structure_file}_{appearance_file}.png',
                'image/png', final_image_endpoint,
                session=session)

    # -- when done, delete all images?
    # -- the instance will be deleted anyway
    pass


def run_model(args):
    ii2s = Embedding(args)
    #
    # ##### Option 1: input folder
    # # ii2s.invert_images_in_W()
    # # ii2s.invert_images_in_FS()

    # ##### Option 2: image path
    # # ii2s.invert_images_in_W('input/face/28.png')
    # # ii2s.invert_images_in_FS('input/face/28.png')
    #
    # #### Option 3: image path list

    # im_path1 = 'input/face/90.png'
    # im_path2 = 'input/face/15.png'
    # im_path3 = 'input/face/117.png'

    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_path2 = os.path.join(args.input_dir, args.im_path2)
    im_path3 = os.path.join(args.input_dir, args.im_path3)

    im_set = {im_path1, im_path2, im_path3}
    ii2s.invert_images_in_W([*im_set])
    ii2s.invert_images_in_FS([*im_set])

    align = Alignment(args)
    align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)
    if im_path2 != im_path3:
        align.align_images(im_path1, im_path3, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)

    blend = Blending(args)
    blend.blend_images(im_path1, im_path2, im_path3, sign=args.sign)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Barbershop')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images')
    parser.add_argument('--im_path1', type=str, default='90.png', help='Identity image')
    parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
    parser.add_argument('--im_path3', type=str, default='15.png', help='Appearance image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')
    parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')

    # Embedding loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
    parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
    parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
    parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')

    # Alignment loss options
    parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
    parser.add_argument('--align_steps1', type=int, default=140, help='')
    parser.add_argument('--align_steps2', type=int, default=100, help='')

    # Blend loss options
    parser.add_argument('--face_lambda', type=float, default=1.0, help='')
    parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
    parser.add_argument('--blend_steps', type=int, default=400, help='')

    # host = args.api_host
    # record_id = args.api_record_id
    # token = args.api_access_token
    parser.add_argument('--api_host', type=str)
    parser.add_argument('--api_record_id', type=str)
    parser.add_argument('--api_access_token', type=str)

    parsed_args = parser.parse_args()
    main(parsed_args)