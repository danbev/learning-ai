import Foundation
import Metal

// Use @main attribute for Swift 5.3+ or this explicit main for older Swift versions
// This works better with Swift Package Manager in some configurations
@main
struct SimpleMetalApp {
    static func main() {
        // Metal (MTL) Device
        let devices = MTLCopyAllDevices()
        var device: MTLDevice? = nil
        
        for availableDevice in devices {
            if availableDevice.name == "Apple M3" {
                device = availableDevice
                print("Using Metal device: \(availableDevice.name)")
                break
            }
        }
        
        guard let device = device else {
            print("Apple M3 GPU is not available.")
            exit(-1)
        }
        
        do {
            let libraryPath = "kernel.metallib"
            guard let defaultLibrary = try? device.makeLibrary(filepath: libraryPath) else {
                print("Failed to load the library. Make sure kernel.metallib is in the current directory.")
                exit(-1)
            }
            
            print("Functions in library:")
            for name in defaultLibrary.functionNames {
                print("    \(name)")
            }
            
            guard let kernelFunction = defaultLibrary.makeFunction(name: "simpleMultiply") else {
                print("Failed to find the kernel function.")
                exit(-1)
            }
            
            print("Kernel function: \(kernelFunction.name)")
            
            let computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
            let commandQueue = device.makeCommandQueue()!
            
            let dataSize = 1024 // number of float elements
            var inputData = [Float](repeating: 0, count: dataSize)
            var outputData = [Float](repeating: 0, count: dataSize)
            
            // Initialize input data
            for i in 0..<dataSize {
                inputData[i] = Float(i)
            }
            
            // Create Metal buffers
            let inputBuffer = device.makeBuffer(bytes: inputData, length: dataSize * MemoryLayout<Float>.size, options: .storageModeShared)!
            let outputBuffer = device.makeBuffer(length: dataSize * MemoryLayout<Float>.size, options: .storageModeShared)!
            
            // Create command buffer and encoder
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            
            computeEncoder.setComputePipelineState(computePipelineState)
            computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            
            let gridSize = MTLSize(width: dataSize, height: 1, depth: 1)
            var threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup
            if threadGroupSize > dataSize {
                threadGroupSize = dataSize
            }
            let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
            
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Copy output data back to the host
            let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: dataSize)
            outputData = Array(UnsafeBufferPointer(start: outputPtr, count: dataSize))
            
            // Uncomment to print results
            // for i in 0..<dataSize {
            //     print("Output[\(i)] = \(outputData[i])")
            // }
            
        } catch {
            print("Error: \(error.localizedDescription)")
            exit(-1)
        }
    }
}
