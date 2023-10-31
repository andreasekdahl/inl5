//
//  ContentView.swift
//  inl4
//
//  Created by Andreas Ekdahl on 2023-10-22.
//

import SwiftUI

import Vision
import Foundation
import UIKit

struct ContentView: View {
    @State private var resultText = ""
    let images = ["image1", "cat","elephant224"]
//    let images = ["AfricanBushElephant"]

    @State private var selectedItem: (index: String, match: (String, Double))?
    var body: some View {
        List {
            ForEach(0..<images.count, id: \.self) { index in
                HStack {
                    Image(images[index])
                        .resizable()
                        .frame(width: 50, height: 50)
                    
                    Button(action: {
                        selectedItem = (images[index],self.doImage(images[index]))
                    }) {
                        Text("Animal match")
                    }
                    .padding()
                }
            }
        }
        
        if let selectedItem = selectedItem {
            HStack{
                VStack(alignment: .leading) {
                    Text("Matching: \(selectedItem.match.0)")
                    Text("Probability: \(selectedItem.match.1)")
                }
            }
            .font(.system(size: 20, weight: .regular))
            
        }
    }

    func doImage(_ imageName: String) -> (String, Double) {
        
        let defaultConfig = MLModelConfiguration()
            
        defaultConfig.computeUnits = .cpuOnly
        
        // Create an instance of the image classifier's wrapper class.

        let imageClassifierWrapper = try? CatAndElephantClassification(configuration: defaultConfig)

        let theimage = UIImage(named: imageName)

        let theimageBuffer = buffer(from: theimage!)!

        do {
            let output = try imageClassifierWrapper!.prediction(image: theimageBuffer)
            return (output.classLabel, output.classLabelProbs[output.classLabel]!)
        } catch {
            // Handle any other unexpected errors
            print("Unexpected Error: \(error)")
        }
        return("No match", 0)
    }

    func buffer(from image: UIImage) -> CVPixelBuffer? {
      let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
      var pixelBuffer : CVPixelBuffer?
      let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
      guard (status == kCVReturnSuccess) else {
        return nil
      }

      CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
      let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

      context?.translateBy(x: 0, y: image.size.height)
      context?.scaleBy(x: 1.0, y: -1.0)

      UIGraphicsPushContext(context!)
      image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
      UIGraphicsPopContext()
      CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

      return pixelBuffer
    }
}

#Preview {
    ContentView()
}
