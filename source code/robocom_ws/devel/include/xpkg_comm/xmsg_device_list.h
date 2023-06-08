// Generated by gencpp from file xpkg_comm/xmsg_device_list.msg
// DO NOT EDIT!


#ifndef XPKG_COMM_MESSAGE_XMSG_DEVICE_LIST_H
#define XPKG_COMM_MESSAGE_XMSG_DEVICE_LIST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <xpkg_comm/xmsg_device.h>

namespace xpkg_comm
{
template <class ContainerAllocator>
struct xmsg_device_list_
{
  typedef xmsg_device_list_<ContainerAllocator> Type;

  xmsg_device_list_()
    : dev_count(0)
    , dev_list()  {
    }
  xmsg_device_list_(const ContainerAllocator& _alloc)
    : dev_count(0)
    , dev_list(_alloc)  {
  (void)_alloc;
    }



   typedef uint8_t _dev_count_type;
  _dev_count_type dev_count;

   typedef std::vector< ::xpkg_comm::xmsg_device_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::xpkg_comm::xmsg_device_<ContainerAllocator> >::other >  _dev_list_type;
  _dev_list_type dev_list;





  typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> const> ConstPtr;

}; // struct xmsg_device_list_

typedef ::xpkg_comm::xmsg_device_list_<std::allocator<void> > xmsg_device_list;

typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_list > xmsg_device_listPtr;
typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_list const> xmsg_device_listConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::xpkg_comm::xmsg_device_list_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace xpkg_comm

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'xpkg_comm': ['/home/eaibot/robocom_ws/src/xpkg_comm/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2ceb7f0b4db86b40356e12aab2f69511";
  }

  static const char* value(const ::xpkg_comm::xmsg_device_list_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2ceb7f0b4db86b40ULL;
  static const uint64_t static_value2 = 0x356e12aab2f69511ULL;
};

template<class ContainerAllocator>
struct DataType< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "xpkg_comm/xmsg_device_list";
  }

  static const char* value(const ::xpkg_comm::xmsg_device_list_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 dev_count\n\
xmsg_device[] dev_list\n\
\n\
================================================================================\n\
MSG: xpkg_comm/xmsg_device\n\
uint8 dev_class\n\
uint8 dev_type\n\
uint8 dev_number\n\
uint8 dev_enable\n\
";
  }

  static const char* value(const ::xpkg_comm::xmsg_device_list_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.dev_count);
      stream.next(m.dev_list);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct xmsg_device_list_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::xpkg_comm::xmsg_device_list_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::xpkg_comm::xmsg_device_list_<ContainerAllocator>& v)
  {
    s << indent << "dev_count: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dev_count);
    s << indent << "dev_list[]" << std::endl;
    for (size_t i = 0; i < v.dev_list.size(); ++i)
    {
      s << indent << "  dev_list[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::xpkg_comm::xmsg_device_<ContainerAllocator> >::stream(s, indent + "    ", v.dev_list[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // XPKG_COMM_MESSAGE_XMSG_DEVICE_LIST_H
