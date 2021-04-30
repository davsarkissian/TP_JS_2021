require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs-extra');
const jpeg = require('jpeg-js');
const Bromise = require('bluebird');
const R = require('ramda');
const sizeOf = require('image-size');
const path = require('path');
const Datastore = require('nedb');

const source = path.resolve('img/ois2.jpg');
const readJpg = async (path) => jpeg.decode(await fs.readFile(path), true);
(async () => {
  const imgList = await Bromise.map([source], readJpg);
  const model = await cocoSsd.load();
  const predictions = await Bromise.map(imgList, (x) => model.detect(x));
  // Console.log('Predictions: ');
  identification = R.flatten(predictions);
  console.log(identification);


  const get_class = R.pipe(R.pluck('class'), R.nth(0));
  const get_score = R.pipe(R.pluck('score'), R.nth(0));


  const createDir = fs.ensureDir;
  const create_directory = R.pipe(R.pluck('class'), R.map(createDir));
  create_directory(identification);


  const check_nbfichier = R.pipe(fs.readdir, R.andThen(R.length));
  const get_destination = async (d, p) => './' + d + '/' + d +
    (await check_nbfichier(d)) + '_' + get_score(p).toFixed(2) + '.jpg';
  
  const destination = await get_destination(
    get_class(identification),
    identification
  );
  
  const relocate_directory = (src, dest) => fs.move(src, dest);
  const deplace = R.pipe(relocate_directory);
  deplace(source, destination);
  


  const get_bbox = R.pipe(R.pluck('bbox'), R.nth(0));
  
  const get_size_w = (source) => sizeOf(source).width;
  const get_size_h = (source) => sizeOf(source).height;
  
  const divh = (x) => x / get_size_h(source);
  const divw = (x) => x / get_size_w(source);
  
  const rounding = (x) => x.toFixed(4);
  
  const get_bbox_unit_scale = R.pipe(
    get_bbox,
    /* Nous n'avons réussi à récupérer les **index** du tableau bbox
        R.map(
            R.cond([
                [R.modulo(**index**,2), R.divide(R.__,get_size_h(image))],
                [R.T, R.divide(R.__,get_size_w(image))]
            ]))
        */
    R.adjust(0, divw, R.__),
    R.adjust(1, divh, R.__),
    R.adjust(2, divw, R.__),
    R.adjust(3, divh, R.__),
    R.map(rounding)
  );
  

  const file_csv = './file_all_animals.csv';
  const init_csv = (file) => new Datastore({filename: file});
  const db = init_csv(file_csv);
  
  const load_data = db.loadDatabase();
  load_data;
  
  const insert_data = (data) => db.insert(data);
  
  const animalID =
    get_class(identification) +
    (await check_nbfichier(get_class(identification)));
  const info_animal = (a, b, c, d) => ({_id: a, breed: b, bbox: c, path: d});
  
  const csv_fill = R.pipe(info_animal, insert_data);
  csv_fill(
    animalID,
    get_class(identification),
    get_bbox_unit_scale(identification),
    destination
  );

  console.log('done');
})();
