//using OpenCvSharp;
using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Dnn;
using Emgu.CV.Util;

namespace YunetFace
{

    internal class Program
    {
        public static void Visualize(Mat input, List<float[]> faces, int thickness = 2)
        {
            if (faces[1] != null)
            {
                for (int idx = 0; idx < faces.Count; idx++)
                {
                    float[] face = faces[idx];
                    Console.WriteLine("Face {0}, top-left coordinates: ({1}, {2}), box width: {3}, box height {4}, score: {5:0.00}", idx, face[0], face[1], face[2], face[3], face[4]);

                    var coords = face.Take(face.Length - 1).Select(coord => (int)coord).ToArray();
                    CvInvoke.Rectangle(input, new Rectangle(coords[0], coords[1], coords[2], coords[3]), new MCvScalar(0, 255, 0), thickness);
                    CvInvoke.Circle(input, new Point(coords[4], coords[5]), 2, new MCvScalar(255, 0, 0), thickness);
                    CvInvoke.Circle(input, new Point(coords[6], coords[7]), 2, new MCvScalar(0, 0, 255), thickness);
                    CvInvoke.Circle(input, new Point(coords[8], coords[9]), 2, new MCvScalar(0, 255, 0), thickness);
                    CvInvoke.Circle(input, new Point(coords[10], coords[11]), 2, new MCvScalar(255, 0, 255), thickness);
                    CvInvoke.Circle(input, new Point(coords[12], coords[13]), 2, new MCvScalar(0, 255, 255), thickness);
                }
            }
            CvInvoke.PutText(input, string.Format("FPS: {0:0.00}"), new Point(1, 16), FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), thickness);
        }

        private static void Main()
        {
            var modelPath1 = "face_detection_yunet_2023mar.onnx";
            var modelPath2 = "face_recognition_sface_2021dec.onnx";
            var scoreThreshold = 0.5f;
            var nmsThreshold = 0.3f;
            var topK = 5000;
            var size = new Size(320, 320);

            //initiliaze_FaceDetectorYN
            FaceDetectorYN detector = new FaceDetectorYN(modelPath1, "", size, scoreThreshold, nmsThreshold, topK, 0, 0);


            string image1 = "image1.jpg";
            string image2 = "image2.jpg";
            var scale = 1.0f;

            if (!string.IsNullOrEmpty(image1))
            {
                Mat img1 = CvInvoke.Imread(image1);
                int img1Width = (int)(img1.Width * scale);
                int img1Height = (int)(img1.Height * scale);

                //Mat resizedImg1 = new Mat();
                //Size newSize = new Size(img1Width, img1Height);
                //CvInvoke.Resize(img1, resizedImg1, newSize);

                CvInvoke.Resize(img1, img1, new Size(img1Width, img1Height));
                detector.InputSize = new Size(img1Width, img1Height);

                Mat faces1 = new Mat();
                detector.Detect(img1, faces1);

                if (!string.IsNullOrEmpty(image2))
                {
                    Mat img2 = CvInvoke.Imread(image2);
                    detector.InputSize = new Size(img2.Width, img2.Height);
                    Mat faces2 = new Mat();
                    detector.Detect(img2, faces2);

                    FaceRecognizerSF recognizer = new FaceRecognizerSF(modelPath2, "", 0, Target.Cpu);




                }









            }
        }



    }

}