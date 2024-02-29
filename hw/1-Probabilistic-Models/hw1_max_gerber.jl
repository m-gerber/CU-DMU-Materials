import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

# Here is a functional but incorrect answer for the programming question
function f(a, bs)
    max = a*bs[1]
    for i in 2:size(bs)[1]
        temp = a*bs[i]
        for j in 1:length(temp)
            if temp[j] > max[j]
                max[j] = temp[j]
            end
        end
    end
    return max
end

# This is how you create the json file to submit
HW1.evaluate(f, "mage7128@colorado.edu")