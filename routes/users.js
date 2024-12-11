const express = require('express');
const router = express.Router();
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const upload = multer({
  limits: {
    fileSize: 10 * 1024 * 1024, // Set file size limit to 10 MB
  },
});

router.get('/', function(req, res, next) {
  res.render("user/testpage");
});

router.post('/find_face', async (req, res) => {
  try {
      const { image } = req.body;

      if (!image) {
          return res.status(400).send('No image data received');
      }

      const base64Data = image.replace(/^data:image\/jpeg;base64,/, '');

      const python = spawn('python', ['classifier/classifier_file.py'], {
          stdio: ['pipe', 'pipe', 'pipe'],
      });

      python.stdin.write(base64Data);
      python.stdin.end();

      let result = '';

      python.stdout.on('data', function (data) {
          result += data.toString();
      });

      python.stderr.on('data', function (data) {
          console.error(`stderr: ${data}`);
      });

      python.on('exit', function (code) {
          if (code === 0) {
              res.send({ prediction: result.trim() });
          } else {
              console.error('Error in Python script execution');
              res.status(500).send('Error processing the image');
          }
      });
  } catch (err) {
      console.error("Error in processing face", err);
      res.status(500).send('Something went wrong');
  }
});

router.get('/enroll_face', function(req, res, next) {
  res.render("user/enroll_page");
});

router.post('/save_enrolled_image', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) {
      return res.status(400).send('No image data received');
    }

    const base64Data = image.replace(/^data:image\/jpeg;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    const imageFolder = path.join(__dirname, '..', 'classifier', 'camera_enroll_image_set');
    if (!fs.existsSync(imageFolder)) {
      fs.mkdirSync(imageFolder); // Create folder if it doesn't exist
    }

    // Generate a unique file name for each image
    const timestamp = Date.now();
    const filePath = path.join(imageFolder, `image_${timestamp}.jpg`);

    // Save the image to the file system
    fs.writeFileSync(filePath, imageBuffer);

    res.send({ message: 'Image saved successfully', filePath });
  } catch (err) {
    console.error('Error saving image:', err);
    res.status(500).send('Error saving the image');
  }
});

router.post('/process_images', async (req, res) => {
  try {
    const python = spawn('python', ['classifier/enroller.py'], {
      stdio: ['inherit', 'pipe', 'pipe'],
    });

    let output = '';
    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });

    python.on('exit', (code) => {
      if (code === 0) {
        res.send({ message: 'Model trained successfully' });
      } else {
        res.status(500).send('Error in training model');
      }
    });
  } catch (err) {
    console.error('Error processing images:', err);
    res.status(500).send('Error processing the images');
  }
});

router.get('/home_page', function(req, res, next) {
  res.render("user/home_page");
});


module.exports = router;
