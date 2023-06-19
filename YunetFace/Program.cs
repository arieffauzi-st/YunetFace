using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System.Drawing;
using Emgu.CV.Util;

namespace YunetFace
{

    internal class Program
    {
        
        public void Visualize(Mat input, Mat faces, int thickness = 2)
        {
            if (faces != null && faces.Rows > 0)
            {
                float[] faceData = (float[])faces.GetData(jagged: false);

                for (int idx = 0; idx < faces.Rows; idx++)
                {
                    Console.WriteLine("Face {0}, top-left coordinates: ({1}, {2}), box width: {3}, box height: {4}, score: {5}",
                        idx, faceData[idx * faces.Cols], faceData[idx * faces.Cols + 1], faceData[idx * faces.Cols + 2], faceData[idx * faces.Cols + 3], faceData[idx * faces.Cols + 14]);

                    int x = (int)faceData[idx * faces.Cols];
                    int y = (int)faceData[idx * faces.Cols + 1];
                    int width = (int)faceData[idx * faces.Cols + 2];
                    int height = (int)faceData[idx * faces.Cols + 3];

                    CvInvoke.Rectangle(input, new Rectangle(x, y, width, height), new MCvScalar(0, 255, 0), thickness);
                    CvInvoke.Circle(input, new Point((int)faceData[idx * faces.Cols + 4], (int)faceData[idx * faces.Cols + 5]), 2, new MCvScalar(255, 0, 0), thickness);
                    CvInvoke.Circle(input, new Point((int)faceData[idx * faces.Cols + 6], (int)faceData[idx * faces.Cols + 7]), 2, new MCvScalar(0, 0, 255), thickness);
                    CvInvoke.Circle(input, new Point((int)faceData[idx * faces.Cols + 8], (int)faceData[idx * faces.Cols + 9]), 2, new MCvScalar(0, 255, 0), thickness);
                    CvInvoke.Circle(input, new Point((int)faceData[idx * faces.Cols + 10], (int)faceData[idx * faces.Cols + 11]), 2, new MCvScalar(255, 0, 255), thickness);
                    CvInvoke.Circle(input, new Point((int)faceData[idx * faces.Cols + 12], (int)faceData[idx * faces.Cols + 13]), 2, new MCvScalar(0, 255, 255), thickness);
                }
            }

            CvInvoke.PutText(input, $"FPS: ", new Point(1, 16), FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 0), 2);
        }

        private static void Main()
        {
            var program = new Program();
            var modelPath1 = "face_detection_yunet_2022mar.onnx";
            var modelPath2 = "face_recognition_sface_2021dec.onnx";
            var scoreThreshold = 0.5f;
            var nmsThreshold = 0.3f;
            var topK = 5000;
            var size = new Size(320, 320);

            //initiliaze_FaceDetectorYN
            FaceDetectorYN detector = new FaceDetectorYN(modelPath1, "", size, scoreThreshold, nmsThreshold, topK, 0, 0);

            string image1 = "";
            string image2 = "";
            string videoPath = "";
            int deviceId;
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

                    // Align faces
                    float[] faceData1 = (float[])faces1.GetData(jagged: false);
                    float[] faceData2 = (float[])faces2.GetData(jagged: false);

                    float x1 = faceData1[0];
                    float y1 = faceData1[1];
                    float width1 = faceData1[2];
                    float height1 = faceData1[3];

                    float x2 = faceData2[0];
                    float y2 = faceData2[1];
                    float width2 = faceData2[2];
                    float height2 = faceData2[3];

                    Rectangle faceRegion1 = new Rectangle((int)x1, (int)y1, (int)width1, (int)height1);
                    Rectangle faceRegion2 = new Rectangle((int)x2, (int)y2, (int)width2, (int)height2);

                    Mat aligned_face1 = new Mat(img1, faceRegion1);
                    Mat face2_align = new Mat(img2, faceRegion2);

                    // Run feature extraction with given aligned_face
                    Mat feature1 = new Mat();
                    Mat feature2 = new Mat();
                    recognizer.Feature(aligned_face1, feature1);
                    feature1 = feature1.Clone();
                    recognizer.Feature(face2_align, feature2);
                    feature2 = feature2.Clone();

                    // Calculate cosine score
                    double cos_score = recognizer.Match(feature1, feature2, FaceRecognizerSF.DisType.Cosine);

                    // Calculate L2 score
                    double l2_score = recognizer.Match(feature1, feature2, FaceRecognizerSF.DisType.NormL2);

                    double cosine_similarity_threshold = 0.363;
                    double l2_similarity_threshold = 1.128;

                    string msg = "different identities";
                    if (cos_score >= cosine_similarity_threshold)
                    {
                        msg = "the same identity";
                    }
                    Console.WriteLine("They have {0}. Cosine Similarity: {1}, threshold: {2} (higher value means higher similarity, max 1.0).", msg, cos_score, cosine_similarity_threshold);

                    msg = "different identities";
                    if (l2_score <= l2_similarity_threshold)
                    {
                        msg = "the same identity";
                    }
                    Console.WriteLine("They have {0}. NormL2 Distance: {1}, threshold: {2} (lower value means higher similarity, min 0.0).", msg, l2_score, l2_similarity_threshold);
                }
                CvInvoke.WaitKey(0);
            }
            else
            {
                if (!string.IsNullOrEmpty(videoPath))
                {
                    deviceId = Convert.ToInt32(videoPath);
                }
                else
                {
                    deviceId = 0;
                }

                VideoCapture cap = new VideoCapture(deviceId);
                int frameWidth = (int)(cap.Get(CapProp.FrameWidth) * scale);
                int frameHeight = (int)(cap.Get(CapProp.FrameHeight) * scale);
                detector.InputSize = new Size(frameWidth, frameHeight);
                while (CvInvoke.WaitKey(1) < 0)
                {
                    Mat frame = new Mat();
                    Mat faces = new Mat();

                    if (!cap.Read(frame))
                    {
                        Console.WriteLine("No frames grabbed!");
                        break;
                    }
                    CvInvoke.Resize(frame, frame, new Size(frameWidth, frameHeight));
                    program.Visualize(frame, faces);
                    detector.Detect(frame, faces);

                    if (faces != null && faces.Rows > 0)
                    {
                        program.Visualize(frame, faces);
                    }

                    CvInvoke.Imshow("Live", frame);
                }
                CvInvoke.DestroyAllWindows();
            }
        }
    }
}

  
